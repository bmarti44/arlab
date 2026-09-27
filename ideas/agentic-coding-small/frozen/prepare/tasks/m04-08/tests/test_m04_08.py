import pytest
from tag_query import normalize_tag, parse_query
from note_index import NoteIndex


def test_parser_groups_and_normalization():
    assert normalize_tag(' Work-2_x ') == 'work-2_x'
    assert parse_query('Work|HOME urgent -Archived -archived work|work') == (
        [frozenset({'work', 'home'}), frozenset({'urgent'}), frozenset({'work'})],
        frozenset({'archived'}))
    assert parse_query('  \t ') == ([], frozenset())


def test_parser_rejects_malformed_tokens():
    assert parse_query('a-b -c_d') == ([frozenset({'a-b'})], frozenset({'c_d'}))
    for query in ['a|', '|a', 'a||b', '-', '-a|b', '--a', 'a,b', '(a)', 'é']:
        with pytest.raises(ValueError):
            parse_query(query)
    for tag in ['', '-a', 'a b', 'a|b', 'é']:
        with pytest.raises(ValueError):
            normalize_tag(tag)


def test_and_or_exclusions_and_sorting():
    index = NoteIndex()
    index.upsert('c', 'three', ['home', 'urgent', 'archived'])
    index.upsert('b', 'two', ['home', 'urgent'])
    index.upsert('a', 'one', ['work', 'urgent'])
    index.upsert('d', 'four', ['work'])
    assert index.search('Work|HOME urgent -archived') == [
        ('a', 'one', ('urgent', 'work')), ('b', 'two', ('home', 'urgent'))]
    assert index.search('missing|work -missing') == [
        ('a', 'one', ('urgent', 'work')), ('d', 'four', ('work',))]
    assert index.search('missing') == []
    assert index.search('work -work') == []


def test_empty_and_only_negative_queries():
    index = NoteIndex()
    index.upsert('b', 'bare', [])
    index.upsert('a', 'tagged', ['x'])
    assert index.search('') == [('a', 'tagged', ('x',)), ('b', 'bare', ())]
    assert index.search('-x') == [('b', 'bare', ())]
    assert index.search('-absent') == index.search('')
    assert NoteIndex().search('x|y') == []


def test_replacement_removes_stale_memberships():
    index = NoteIndex()
    assert index.upsert('a', 'old', ['Work', 'work', 'OLD']) is None
    index.upsert('b', 'other', ['work'])
    assert index.tag_counts() == {'work': 2, 'old': 1}
    index.upsert('a', 'new', ['new'])
    assert index.search('old') == []
    assert index.search('work') == [('b', 'other', ('work',))]
    assert index.search('new') == [('a', 'new', ('new',))]
    assert index.tag_counts() == {'work': 1, 'new': 1}
    index.upsert('a', 'empty', [])
    assert index.tag_counts() == {'work': 1}


def test_remove_and_missing_id():
    index = NoteIndex()
    index.upsert('a', 'one', ['x'])
    index.upsert('b', 'two', ['x', 'y'])
    assert index.remove('b') is None
    assert index.tag_counts() == {'x': 1}
    with pytest.raises(KeyError):
        index.remove('b')
    assert index.search('') == [('a', 'one', ('x',))]
    index.remove('a')
    assert index.tag_counts() == {} and index.search('') == []


def test_failed_replacement_is_atomic():
    index = NoteIndex()
    index.upsert('a', 'old', ['safe'])
    with pytest.raises(ValueError):
        index.upsert('a', 'new', iter(['new', 'bad tag']))
    with pytest.raises(ValueError):
        index.upsert('b', 'bad', [''])
    assert index.search('') == [('a', 'old', ('safe',))]
    assert index.tag_counts() == {'safe': 1}
    with pytest.raises(ValueError):
        index.search('safe|')


def test_copied_inputs_and_outputs():
    index = NoteIndex()
    tags = ['X', 'x', 'a']
    index.upsert('id', 'text', iter(tags))
    tags.append('later')
    result = index.search('')
    result.clear()
    counts = index.tag_counts()
    counts['x'] = 99
    assert index.search('') == [('id', 'text', ('a', 'x'))]
    assert index.tag_counts() == {'x': 1, 'a': 1}
