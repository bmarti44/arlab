import pytest
from assembler import assemble
from vm import run


def test_assembly_labels_comments_and_targets():
    source = '# header\n start: SET a +2 # count\nJNZ a start\nJMP end\nHALT\nend:'
    assert assemble(source) == [('SET', 'a', 2), ('JNZ', 'a', 0), ('JMP', 4), ('HALT',)]
    assert assemble('x:OUT d\nHALT') == [('OUT', 'd'), ('HALT',)]


def test_arithmetic_registers_and_halt():
    result = run('SET a -3\nADD a +5\nOUT a\nOUT b\nSET d 12345678901234567890\nHALT\nOUT d')
    assert result == {'registers': {'a': 2, 'b': 0, 'c': 0, 'd': 12345678901234567890},
                      'output': [2, 0], 'steps': 6, 'halted': True}


def test_conditional_loop_and_step_count():
    source = 'SET c 3\nloop: OUT c\nADD c -1\nJNZ c loop\nHALT'
    result = run(source, max_steps=11)
    assert result['output'] == [3, 2, 1]
    assert result['registers']['c'] == 0
    assert result['steps'] == 11
    assert result['halted'] is True
    with pytest.raises(RuntimeError):
        run(source, max_steps=10)


def test_jumps_fallthrough_and_empty_program():
    result = run('JMP go\nOUT a\ngo: ADD b 7\nJMP end\nOUT b\nend:', max_steps=3)
    assert result == {'registers': {'a': 0, 'b': 7, 'c': 0, 'd': 0},
                      'output': [], 'steps': 3, 'halted': False}
    assert run(' # empty\nend:')['steps'] == 0
    assert run('')['halted'] is False
    assert run('OUT d', 1)['output'] == [0]
    assert run('HALT', 1)['halted'] is True


def test_invalid_assembly_is_rejected():
    invalid = ['set a 1', 'SET e 1', 'OUT A', 'SET a 1_0', 'ADD a 1.5',
               'HALT x', 'SET a', 'OUT a b', 'JMP nowhere', 'JNZ a missing',
               'x: HALT\nx: OUT a', '1x: HALT', 'a: b: HALT', 'JMP 0',
               'Here: HALT\nJMP here', 'SET a, 1']
    for source in invalid:
        with pytest.raises(ValueError):
            assemble(source)
    with pytest.raises(ValueError):
        run('HALT\nBAD')


def test_step_limit_and_validation():
    with pytest.raises(RuntimeError):
        run('forever: JMP forever', 5)
    with pytest.raises(RuntimeError):
        run('OUT a\nOUT a', 1)
    for limit in [0, -1, True, 1.5, '2']:
        with pytest.raises(ValueError):
            run('HALT', limit)


def test_calls_are_independent_and_labels_can_share_target():
    assert assemble('first:\nsecond:\nOUT a\nJMP first') == [('OUT', 'a'), ('JMP', 0)]
    result = run('SET a 9\nOUT a')
    result['registers']['a'] = 100
    result['output'].append(7)
    assert run('OUT a') == {'registers': {'a': 0, 'b': 0, 'c': 0, 'd': 0},
                           'output': [0], 'steps': 1, 'halted': False}
