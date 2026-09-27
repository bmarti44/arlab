from copy import deepcopy
import pytest
from task_graph import build_graph
from scheduler import schedule


def test_empty_and_single_task():
    assert build_graph([]) == ({}, {})
    assert schedule([]) == dict(order=[], times={}, makespan=0, critical_path=[])
    assert schedule([dict(id="only", duration=7, deps=[])]) == dict(order=["only"], times={"only": (0, 7)}, makespan=7, critical_path=["only"])


def test_parallel_branches_and_critical_chain():
    records = [dict(id="D", duration=2, deps=["B", "C"]), dict(id="C", duration=5, deps=["A"]), dict(id="B", duration=3, deps=["A"]), dict(id="A", duration=2, deps=[])]
    assert schedule(records) == dict(order=["A", "B", "C", "D"], times={"A": (0, 2), "B": (2, 5), "C": (2, 7), "D": (7, 9)}, makespan=9, critical_path=["A", "C", "D"])


def test_ready_queue_competes_immediately():
    records = [dict(id="z", duration=1, deps=[]), dict(id="b", duration=1, deps=["a"]), dict(id="a", duration=1, deps=[])]
    result = schedule(records)
    assert result["order"] == ["a", "b", "z"]
    assert result["times"] == {"a": (0, 1), "b": (1, 2), "z": (0, 1)}
    assert result["makespan"] == 2


def test_local_critical_ties_and_zero_duration():
    records = [dict(id="z", duration=1, deps=[]), dict(id="a", duration=1, deps=[]), dict(id="b", duration=1, deps=["z"]), dict(id="c", duration=1, deps=["a"]), dict(id="end", duration=1, deps=["c", "b"]), dict(id="tail", duration=0, deps=["end"])]
    result = schedule(records)
    assert result["critical_path"] == ["z", "b", "end"]
    assert result["times"]["tail"] == (3, 3)
    zero = schedule([dict(id="b", duration=0, deps=[]), dict(id="a", duration=0, deps=["b"])])
    assert zero == dict(order=["b", "a"], times={"b": (0, 0), "a": (0, 0)}, makespan=0, critical_path=["b", "a"])


def test_normalization_forward_refs_and_isolation():
    records = [dict(id="b", duration=2, deps=["a", "a"], note="ignored"), dict(id="a", duration=1, deps=[])]
    before = deepcopy(records)
    durations, deps = build_graph(records)
    assert durations == {"b": 2, "a": 1}
    assert deps == {"b": {"a"}, "a": set()}
    durations["a"] = 999
    deps["b"].clear()
    assert schedule(records)["times"] == {"a": (0, 1), "b": (1, 3)}
    assert records == before


def test_invalid_graph_records():
    invalid = [
        [{}], [dict(id="", duration=1, deps=[])], [dict(id=1, duration=1, deps=[])],
        [dict(id="a", duration=-1, deps=[])], [dict(id="a", duration=True, deps=[])],
        [dict(id="a", duration=1.5, deps=[])], [dict(id="a", duration=1, deps="b")],
        [dict(id="a", duration=1, deps=[1])], [dict(id="a", duration=1, deps=["missing"])],
        [dict(id="a", duration=1, deps=["a"])],
        [dict(id="a", duration=1, deps=[]), dict(id="a", duration=2, deps=[])],
    ]
    for records in invalid:
        with pytest.raises(ValueError):
            build_graph(records)
        with pytest.raises(ValueError):
            schedule(records)


def test_cycle_rejected_even_with_schedulable_component():
    records = [dict(id="a", duration=1, deps=["b"]), dict(id="b", duration=2, deps=["a"]), dict(id="free", duration=3, deps=[])]
    assert build_graph(records)[1] == {"a": {"b"}, "b": {"a"}, "free": set()}
    with pytest.raises(ValueError):
        schedule(records)


def test_large_layered_graph_uses_graph_algorithm():
    count = 1600
    names = [f"t{i:04d}" for i in range(count)]
    records = [dict(id=name, duration=1, deps=names[max(0, i - 6):i]) for i, name in enumerate(names)]
    result = schedule(list(reversed(records)))
    assert result["order"] == names
    assert result["critical_path"] == names
    assert result["makespan"] == count
    assert result["times"][names[-1]] == (count - 1, count)
