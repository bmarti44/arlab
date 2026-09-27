import pytest
from spooler import PrintSpooler, validate_job


def test_validation_and_ids():
    s = PrintSpooler()
    assert s.submit('first', 1) == 1
    for args in [('', 1, 0), (None, 1, 0), ('x', 0, 0), ('x', True, 0), ('x', 1, True), ('x', 1, 1.5)]:
        with pytest.raises(ValueError):
            validate_job(*args)
        with pytest.raises(ValueError):
            s.submit(*args)
    assert s.submit(' ', 1, -4) == 2


def test_priority_then_fifo():
    s = PrintSpooler()
    ids = [s.submit('low', 1, -1), s.submit('high', 1, 3), s.submit('tie', 1, 3)]
    assert [j['id'] for j in s.snapshot()['queued']] == [2, 3, 1]
    assert [s.tick()['id'] for _ in ids] == [2, 3, 1]
    assert s.tick() is None


def test_nonpreemptive_pages_and_events():
    s = PrintSpooler()
    s.submit('report', 2)
    assert s.tick() == {'id': 1, 'name': 'report', 'page': 1, 'total': 2, 'done': False}
    s.submit('urgent', 1, 100)
    assert s.tick() == {'id': 1, 'name': 'report', 'page': 2, 'total': 2, 'done': True}
    assert s.snapshot()['active'] is None
    assert s.tick()['id'] == 2


def test_reprioritize_preserves_submission_order():
    s = PrintSpooler()
    s.submit('first', 1, 0)
    s.submit('second', 1, 4)
    assert s.reprioritize(1, 4) is True
    assert s.tick()['id'] == 1
    assert s.reprioritize(1, 0) is False
    assert s.reprioritize(99, 0) is False
    assert s.tick()['id'] == 2


def test_cancel_queued_and_active():
    s = PrintSpooler()
    s.submit('a', 4)
    s.submit('b', 1)
    s.submit('c', 1)
    assert s.cancel(2) is True
    assert s.tick()['page'] == 1
    assert s.cancel(1) is True
    assert s.cancel(1) is False
    assert s.cancel(99) is False
    assert s.tick()['id'] == 3
    assert s.cancel(3) is False
    assert s.tick() is None


def test_snapshot_copies_and_active_priority():
    s = PrintSpooler()
    s.submit('a', 2, 3)
    s.submit('b', 1)
    s.tick()
    snap = s.snapshot()
    assert snap['active'] == {'id': 1, 'name': 'a', 'pages': 2, 'priority': 3, 'printed': 1}
    assert snap['queued'] == [{'id': 2, 'name': 'b', 'pages': 1, 'priority': 0, 'printed': 0}]
    snap['active']['printed'] = 200
    snap['queued'][0]['pages'] = 200
    snap['queued'].clear()
    assert s.reprioritize(1, 999) is False
    assert s.tick()['done'] is True
    assert s.tick()['done'] is True


def test_invalid_reprioritize_and_empty():
    s = PrintSpooler()
    assert s.snapshot() == {'active': None, 'queued': []}
    assert s.tick() is None
    s.submit('a', 2)
    before = s.snapshot()
    for job_id in (1, 88):
        with pytest.raises(ValueError):
            s.reprioritize(job_id, False)
    assert s.snapshot() == before
