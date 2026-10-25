"""Resolve explicit provider parameters without writing credentials to Codex Home."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit


def read_env(path):
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key = key.strip()
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
            raise ValueError(f'Invalid env assignment on line {number}')
        tokens = shlex.split(value, comments=True)
        if len(tokens) > 1:
            raise ValueError(f'Quote env values containing spaces on line {number}')
        values[key] = tokens[0] if tokens else ''
    return values


@dataclass
class Settings:
    repo: Path
    env_file: Path
    codex_home: Path
    model: str
    effort: str
    base_url: str
    api_key: str = field(repr=False)
    codex: str = 'codex'
    gpu: int = 0

    def __post_init__(self):
        if self.gpu not in (0, 1):
            raise ValueError('Only physical simulator GPUs 0 and 1 are permitted')
        if not all((self.model, self.effort, self.base_url, self.api_key)):
            raise ValueError('Model, effort, API base and API key must be nonempty')
        url = urlsplit(self.base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('API base must be an HTTP(S) URL without credentials, query or fragment')

    def public(self):
        return {key: str(getattr(self, key)) for key in
                ('repo', 'env_file', 'codex_home', 'model', 'effort', 'base_url', 'codex')} | {'gpu': self.gpu}

    def command(self):
        overrides = {
            'model_provider': 'scene_maps', 'model': self.model,
            'model_reasoning_effort': self.effort,
            'model_providers.scene_maps.name': 'Scene map provider',
            'model_providers.scene_maps.base_url': self.base_url,
            'model_providers.scene_maps.env_key': 'SCENE_MAP_API_KEY',
            'model_providers.scene_maps.wire_api': 'responses',
            'model_providers.scene_maps.requires_openai_auth': False,
            'model_providers.scene_maps.supports_websockets': False,
            'model_providers.scene_maps.request_max_retries': 5,
            'model_providers.scene_maps.stream_max_retries': 5,
            'approval_policy': 'never', 'sandbox_mode': 'danger-full-access',
            'shell_environment_policy.inherit': 'core',
            'shell_environment_policy.exclude': ['*KEY*', '*TOKEN*', '*SECRET*', '*PASSWORD*'],
            'shell_environment_policy.set.CUDA_VISIBLE_DEVICES': str(self.gpu),
            'shell_environment_policy.set.__EGL_VENDOR_LIBRARY_DIRS': '/usr/share/glvnd/egl_vendor.d',
        }
        command = [self.codex, 'app-server', '--listen', 'stdio://']
        for key, value in overrides.items():
            command.extend(['-c', f'{key}={json.dumps(value)}'])
        return command

    def environment(self):
        env = dict(os.environ)
        env.update(CODEX_HOME=str(self.codex_home), SCENE_MAP_API_KEY=self.api_key,
                   CUDA_VISIBLE_DEVICES=str(self.gpu),
                   __EGL_VENDOR_LIBRARY_DIRS='/usr/share/glvnd/egl_vendor.d')
        return env
