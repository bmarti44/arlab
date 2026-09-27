import pytest
from isbn import check_digit10, check_digit13, is_valid, normalize, to_isbn13


def test_ten_digit_check_values():
    assert check_digit10('030640615') == '2'
    assert check_digit10('080442957') == 'X'
    assert check_digit10('000000000') == '0'


def test_thirteen_digit_check_values():
    assert check_digit13('978030640615') == '7'
    assert check_digit13('978080442957') == '3'
    assert check_digit13('000000000000') == '0'


def test_valid_shapes_and_invalid_checks():
    assert is_valid('0-306-40615-2')
    assert is_valid('9780306406157')
    assert is_valid('0 8044 2957 x')
    assert not is_valid('0306406153')
    assert not is_valid('9780306406158')
    assert is_valid('0000000000')
    assert is_valid('0000000000000')


def test_normalization_and_conversion():
    assert normalize(' 0-8044-2957-x ') == '080442957X'
    assert to_isbn13('0306406152') == '9780306406157'
    assert to_isbn13('080442957x') == '9780804429573'
    assert to_isbn13('978-0-306-40615-7') == '9780306406157'


def test_malformed_and_type_errors():
    assert is_valid('0306406152')
    for text in ['', 'X306406152', '0306406152\t', '９780306406157', '978030640615X']:
        assert is_valid(text) is False
        with pytest.raises(ValueError):
            normalize(text)
        with pytest.raises(ValueError):
            to_isbn13(text)
    for function in [check_digit10, check_digit13, is_valid, normalize, to_isbn13]:
        with pytest.raises(TypeError):
            function(None)


def test_prefix_compaction_and_bad_lengths():
    assert check_digit10('0-306 40615') == '2'
    assert check_digit13('978-0-306-40615') == '7'
    for function, bad in [(check_digit10, '12345678'), (check_digit10, '12345678X'),
                          (check_digit10, '１２３４５６７８９'), (check_digit13, '97803064061'),
                          (check_digit13, '978030640615\n')]:
        with pytest.raises(ValueError):
            function(bad)
