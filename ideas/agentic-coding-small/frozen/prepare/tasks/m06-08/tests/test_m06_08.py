import pytest
from customers import Customer, parse_customers
from bank import simulate, summarize


def test_parse_fields_blanks_and_order():
    assert parse_customers('  late , 010 , 2\n\n early,0,01 \n') == [Customer('late', 10, 2), Customer('early', 0, 1)]
    assert parse_customers(' \n\t') == []


def test_parser_rejects_invalid_records():
    for text in ['a,0', 'a,0,1,extra', ',0,1', 'a,0,1\na,2,3', 'a,-1,2', 'a,+1,2', 'a,1.5,2', 'a,0,0', 'a,０,1', 'a,,1']:
        with pytest.raises(ValueError):
            parse_customers(text)


def test_single_teller_waits_and_summary():
    records = simulate(parse_customers('a,0,4\nb,1,2\nc,7,1'), 1)
    assert records == [
        {'customer_id': 'a', 'teller': 0, 'arrival': 0, 'start': 0, 'end': 4, 'wait': 0},
        {'customer_id': 'b', 'teller': 0, 'arrival': 1, 'start': 4, 'end': 6, 'wait': 3},
        {'customer_id': 'c', 'teller': 0, 'arrival': 7, 'start': 7, 'end': 8, 'wait': 0}]
    assert summarize(records, 1) == {'served': 3, 'total_wait': 3, 'max_wait': 3, 'last_end': 8, 'busy': [7]}


def test_sort_stability_and_input_unchanged():
    customers = [Customer('later', 5, 1), Customer('first', 0, 2), Customer('second', 0, 1)]
    original = list(customers)
    records = simulate(customers, 1)
    assert [r['customer_id'] for r in records] == ['first', 'second', 'later']
    assert [r['start'] for r in records] == [0, 2, 5]
    assert customers == original


def test_teller_ties_and_exact_release():
    records = simulate(parse_customers('a,0,3\nb,0,3\nc,1,1\nd,3,1'), 2)
    assert [(r['teller'], r['start'], r['wait']) for r in records] == [(0, 0, 0), (1, 0, 0), (0, 3, 2), (1, 3, 0)]
    assert summarize(records, 2)['busy'] == [4, 4]


def test_idle_teller_choice_and_last_end():
    records = simulate(parse_customers('a,0,5\nb,0,1\nc,10,1'), 2)
    assert [(r['teller'], r['start']) for r in records] == [(0, 0), (1, 0), (0, 10)]
    other = simulate([Customer('long', 0, 10), Customer('short', 1, 1)], 2)
    assert summarize(other, 2)['last_end'] == 10


def test_empty_and_invalid_teller_count():
    assert simulate([], 3) == []
    assert summarize([], 3) == {'served': 0, 'total_wait': 0, 'max_wait': 0, 'last_end': 0, 'busy': [0, 0, 0]}
    for count in [0, -2]:
        with pytest.raises(ValueError):
            simulate([], count)
        with pytest.raises(ValueError):
            summarize([], count)
