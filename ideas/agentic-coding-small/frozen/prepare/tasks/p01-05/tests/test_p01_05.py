import pytest
from expr_tokens import tokenize
from expression import evaluate

def test_tokenizer_contract():
    assert tokenize(' \t.5 + rate_2*(12.30-7)\n') == [('NUMBER', '.5'), ('+', '+'), ('IDENT', 'rate_2'), ('*', '*'), ('(', '('), ('NUMBER', '12.30'), ('-', '-'), ('NUMBER', '7'), (')', ')'), ('EOF', '')]
    assert tokenize('') == [('EOF', '')]
    assert tokenize('01 _X / 2') == [('NUMBER', '01'), ('IDENT', '_X'), ('/', '/'), ('NUMBER', '2'), ('EOF', '')]
    for text in ('2@3', '1.', 'é', '2,3'):
        with pytest.raises(ValueError):
            tokenize(text)

def test_precedence_and_left_associativity():
    assert evaluate('2 + 3 * 4 - 5') == 9.0
    assert evaluate('20 / 2 / 5') == 2.0
    assert evaluate('10 - 3 - 2') == 5.0
    assert evaluate('(2+3)*(4-1)') == 15.0
    assert type(evaluate('2')) is float

def test_unary_signs_and_fractional_numbers():
    assert evaluate('---2 + +.5') == -1.5
    assert evaluate('2*-(-3 + 1)') == 4.0
    assert evaluate('6/-2') == -3.0
    assert evaluate('1--2') == 3.0
    assert evaluate('.25 + 1.25 / 2') == 0.875

def test_variables_and_unchanged_mapping():
    variables = {'price': 12.5, 'qty': 3, '_discount': 2, 'Price': 1, 'unused': None}
    before = variables.copy()
    assert evaluate('price * qty - _discount + Price', variables) == 36.5
    assert variables == before
    with pytest.raises(KeyError) as exc:
        evaluate('Price + missing', variables)
    assert exc.value.args == ('missing',)
    with pytest.raises(KeyError) as exc:
        evaluate('x')
    assert exc.value.args == ('x',)

def test_invalid_variable_values():
    for value in (True, None, '3', float('inf'), float('nan')):
        with pytest.raises(ValueError):
            evaluate('x + 1', {'x': value})
    assert evaluate('2', {'unused': 'bad'}) == 2.0

def test_syntax_errors():
    for text in ('', '   ', '1 +', '*2', '()', '(2', '2)', '2 3', '2(3)', '1e3', '2**3', '1//2', '.', '1.2.3', 'x=2'):
        with pytest.raises(ValueError):
            evaluate(text)

def test_zero_division_and_nested_expression():
    for text in ('1/0', '2/(3-3)', '1/-0'):
        with pytest.raises(ZeroDivisionError):
            evaluate(text)
    assert evaluate('(' * 30 + 'x+1' + ')' * 30, {'x': 4}) == 5.0
    assert evaluate('+'.join(['1'] * 70)) == 70.0
