from outline import Outline, parse_heading


def test_heading_syntax():
    assert parse_heading('   ## Title ###  ') == (2, 'Title')
    assert parse_heading('# C#') == (1, 'C#')
    assert parse_heading('###\t**bold**') == (3, '**bold**')
    assert parse_heading('#') == (1, '')
    assert parse_heading('# ###') == (1, '')
    for line in ['####### no', '#joined', '    # indented', '\t# tab', 'Title', '===']:
        assert parse_heading(line) is None


def test_hierarchy_and_skipped_levels():
    outline = Outline()
    assert outline.add('### Deep\n# Top\n### Skip\n## Mid\n### Child\n# Next') is None
    assert outline.entries() == [
        (3, '0.0.1', 'Deep'), (1, '1', 'Top'), (3, '1.0.1', 'Skip'),
        (2, '1.1', 'Mid'), (3, '1.1.1', 'Child'), (1, '2', 'Next')]


def test_fence_character_and_length():
    outline = Outline()
    outline.add('# A\n````python\n# hidden\n```\n# hidden too\n~~~~\n# no\n```` \n## B')
    assert outline.render() == '1 A\n1.1 B'


def test_fence_closer_tail_and_tildes():
    outline = Outline()
    outline.add('  ~~~lang\n# hidden\n~~~ still open\n## hidden\n   ~~~~\n# Visible')
    assert outline.entries() == [(1, '1', 'Visible')]


def test_calls_accumulate_but_fences_do_not():
    outline = Outline()
    outline.add('# One\n```\n# hidden')
    outline.add('## Two\n# Three')
    assert outline.render() == '1 One\n1.1 Two\n2 Three'


def test_empty_render_and_reset():
    outline = Outline()
    assert outline.render() == ''
    outline.add('#\n## A')
    assert outline.render() == '1 \n1.1 A'
    assert outline.reset() is None
    assert outline.entries() == []
    outline.add('## Fresh')
    assert outline.render() == '0.1 Fresh'


def test_snapshot_and_nonheadings():
    outline = Outline()
    outline.add('plain\n----\n    # indented\n# A\r\n###### Six')
    snapshot = outline.entries()
    snapshot.clear()
    assert outline.entries() == [(1, '1', 'A'), (6, '1.0.0.0.0.1', 'Six')]
