"""Small stdio JSONL client for the installed Codex app-server."""

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import signal
import time

from src.common.io import redact


class AppServer:
    def __init__(self, command, env, cwd, directory, secrets=()):
        self.command, self.env, self.cwd = command, env, cwd
        self.directory, self.secrets = Path(directory), secrets
        self.process = None
        self.reader = self.stderr_reader = None
        self.pending = {}
        self.notifications = asyncio.Queue()
        self.counter = 0
        self.stderr_tail = deque(maxlen=20)

    def record(self, direction, message):
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'direction': direction,
                                     'message': redact(message, self.secrets)}, ensure_ascii=False) + '\n')

    async def send(self, message):
        self.record('client', message)
        self.process.stdin.write((json.dumps(message) + '\n').encode())
        await self.process.stdin.drain()

    async def request(self, method, params, timeout=90):
        self.counter += 1
        identifier = self.counter
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.send({'id': identifier, 'method': method, 'params': params})
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(identifier, None)

    async def _read(self):
        error = RuntimeError('Codex app-server closed stdout')
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                self.record('server', message)
                if 'id' in message and 'method' in message:
                    # Batch execution cannot answer arbitrary human questions or approvals.
                    await self.send({'id': message['id'], 'error': {
                        'code': -32601, 'message': 'Interactive server requests are unsupported by this batch runner'}})
                    raise RuntimeError(f"Unexpected interactive request: {message['method']}")
                if 'id' in message:
                    future = self.pending.get(message['id'])
                    if future is not None and not future.done():
                        if 'error' in message:
                            future.set_exception(RuntimeError(str(redact(message['error'], self.secrets))))
                        else:
                            future.set_result(message.get('result'))
                else:
                    await self.notifications.put(message)
        except Exception as exc:
            error = exc
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            await self.notifications.put(error)

    async def _stderr(self):
        with (self.directory / 'stderr.log').open('a') as stream:
            while line := await self.process.stderr.readline():
                text = redact(line.decode(errors='replace'), self.secrets)
                self.stderr_tail.append(text)
                stream.write(text)
                stream.flush()

    async def __aenter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.process = await asyncio.create_subprocess_exec(
            *self.command, env=self.env, cwd=self.cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, limit=32 * 1024 * 1024)
        self.reader = asyncio.create_task(self._read())
        self.stderr_reader = asyncio.create_task(self._stderr())
        try:
            await self.request('initialize', {'clientInfo': {'name': 'navcraft_maps',
                'title': 'NavCraft scene map builder', 'version': '0.1.0'}})
            await self.send({'method': 'initialized'})
        except BaseException:
            await self.close()
            raise
        return self

    async def turn(self, thread_id, inputs, model, effort, schema, timeout, on_started=None):
        # Notifications may precede the turn/start reply; the reader queues all of them.
        result = await self.request('turn/start', {'threadId': thread_id, 'input': inputs,
            'model': model, 'effort': effort, 'outputSchema': schema})
        turn_id = result['turn']['id']
        if on_started:
            on_started(turn_id)
        answers = []
        try:
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                notification = await asyncio.wait_for(self.notifications.get(), max(0, remaining))
                if isinstance(notification, Exception):
                    raise notification
                params = notification.get('params', {})
                if params.get('threadId') != thread_id:
                    continue
                method = notification.get('method')
                if method == 'item/completed' and params.get('turnId') == turn_id:
                    item = params.get('item', {})
                    if item.get('type') == 'agentMessage':
                        answers.append(item.get('text', ''))
                if method == 'turn/completed' and params['turn']['id'] == turn_id:
                    turn = params['turn']
                    if turn['status'] != 'completed':
                        raise RuntimeError(f"Turn {turn_id}: {turn['status']}: {turn.get('error')}")
                    return {'turn_id': turn_id, 'answer': answers[-1] if answers else ''}
        except BaseException:
            try:
                await self.request('turn/interrupt', {'threadId': thread_id, 'turnId': turn_id}, timeout=5)
            except Exception:
                pass
            raise

    async def close(self):
        if self.process:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                pass
            # Stop remaining processes in this server's group, never unrelated servers.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
        for task in (self.reader, self.stderr_reader):
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in (self.reader, self.stderr_reader) if t), return_exceptions=True)

    async def __aexit__(self, *args):
        await self.close()
