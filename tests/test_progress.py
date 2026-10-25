from concurrent.futures import ThreadPoolExecutor

from src.common.progress import Progress


def test_waiting_is_not_completion_and_resume_is_monotonic(capsys):
    progress = Progress("Maps", ["a", "b", "c"], {"a": "ready"})
    progress.set("a", "running")
    progress.set("b", "waiting")
    progress.set("c", "partial")
    text = capsys.readouterr().err
    assert '2/3 ended' in text
    assert 'ready=1 partial=1 waiting=1' in text
    assert progress.states['a'] == 'ready'
    progress.set('b', 'failed')
    assert '3/3 ended' in capsys.readouterr().err


def test_parallel_progress_counts_unique_tasks(capsys):
    progress = Progress('Episodes', range(20))
    def complete(index):
        progress.set(index, 'running')
        progress.set(index, 'finished')
        progress.set(index, 'finished')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(complete, range(20)))
    lines = capsys.readouterr().err.splitlines()
    assert '20/20 ended | finished=20' in lines[-1]
    assert len(lines) == 41  # Initial + two transitions per unique task.
