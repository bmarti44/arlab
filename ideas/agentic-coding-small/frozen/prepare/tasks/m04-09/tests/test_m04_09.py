import pytest
from transition_table import parse_table
from automaton import Automaton, format_trace

TABLE = 'state,a,b\nS,A,S\nA,S,-\n'


def test_parse_forward_references_and_missing():
    assert parse_table('\n state, b, a \n S, -, A \n A, S, A\n\n') == (
        ('b', 'a'), {'S': {'a': 'A'}, 'A': {'b': 'S', 'a': 'A'}})


def test_bad_tables():
    assert parse_table('state,a\nS,S')[0] == ('a',)
    bad = ['', 'state,a', 'wrong,a\nS,S', 'state\nS', 'state,a,a\nS,S,S',
           'state,ab\nS,S', 'state,A\nS,S', 'state,a\nS,S,S',
           'state,a\nS,S\nS,S', 'state,a\n1S,1S', 'state,a\nS,missing',
           'state,a\nS,']
    for table in bad:
        with pytest.raises(ValueError):
            parse_table(table)


def test_success_rejection_and_format():
    machine = Automaton(TABLE, 'S', ['S'])
    result = machine.run('aa')
    assert result == {'accepted': True, 'state': 'S', 'trace': ['S', 'A', 'S'], 'consumed': 2, 'error': None}
    assert format_trace(result) == 'S -> A -> S | accepted'
    rejected = machine.run('a')
    assert rejected == {'accepted': False, 'state': 'A', 'trace': ['S', 'A'], 'consumed': 1, 'error': None}
    assert format_trace(rejected) == 'S -> A | rejected'


def test_missing_transition_stops_before_character():
    result = Automaton(TABLE, 'S', ['A']).run('abz')
    assert result == {'accepted': False, 'state': 'A', 'trace': ['S', 'A'], 'consumed': 1, 'error': 'missing transition'}
    assert format_trace(result) == 'S -> A | rejected: missing transition at 1'


def test_unknown_symbol_and_empty_text():
    machine = Automaton(TABLE, 'S', ['S'])
    assert machine.run('') == {'accepted': True, 'state': 'S', 'trace': ['S'], 'consumed': 0, 'error': None}
    result = machine.run('?a')
    assert result == {'accepted': False, 'state': 'S', 'trace': ['S'], 'consumed': 0, 'error': 'unknown symbol'}
    assert format_trace(result) == 'S | rejected: unknown symbol at 0'
    assert machine.run('aa?')['consumed'] == 2
    assert Automaton(TABLE, 'A', []).run('')['accepted'] is False


def test_configuration_errors():
    assert Automaton(TABLE, 'A', ['A']).run('')['accepted'] is True
    with pytest.raises(ValueError):
        Automaton(TABLE, 'Z', [])
    with pytest.raises(ValueError):
        Automaton(TABLE, 'S', ['Z'])
    with pytest.raises(ValueError):
        Automaton('state,a\nS,Z', 'S', [])


def test_runs_and_configuration_are_independent():
    accepting = ['S', 'S']
    machine = Automaton(TABLE, 'S', iter(accepting))
    accepting.clear()
    first = machine.run('aa')
    first['trace'].clear()
    assert machine.run('b') == {'accepted': True, 'state': 'S', 'trace': ['S', 'S'], 'consumed': 1, 'error': None}
    assert machine.run('ab')['error'] == 'missing transition'
    assert machine.run('')['trace'] == ['S']
