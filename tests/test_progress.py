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
    assert progress.bar.disable  # Completed bars release their terminal line.


def test_parallel_progress_counts_unique_tasks(capsys):
    progress = Progress('Episodes', range(20))
    def complete(index):
        progress.set(index, 'running')
        progress.set(index, 'finished')
        progress.set(index, 'finished')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(complete, range(20)))
    text = capsys.readouterr().err
    assert '20/20 ended' in text and 'finished=20' in text
    assert progress.bar.n == 20
    assert 'running=' in text
    assert '\r' in text
    assert text.count('\n') == 1  # Refresh in place; newline only on completion.


def test_empty_and_already_finished_progress_close_immediately(capsys):
    for keys, initial in [([], {}), (['a'], {'a': 'ready'})]:
        progress = Progress('Maps', keys, initial)
        assert progress.bar.disable
    assert capsys.readouterr().err.count('\n') == 2
