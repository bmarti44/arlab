import pytest
from diff_parser import Hunk, parse_patch
from patcher import apply_patch

HEAD = '--- old\n+++ new\n'


def test_parser_representation_and_default_counts():
    patch = HEAD + '@@ -01 +01 @@\n-old\n+new'
    assert parse_patch(patch) == [Hunk(1, 1, 1, 1, (('-', 'old'), ('+', 'new')))]
    assert parse_patch('') == []
    assert parse_patch(HEAD) == []
    assert parse_patch(HEAD.rstrip('\n')) == []


def test_replace_context_and_untouched_suffix():
    patch = HEAD + '@@ -2,3 +2,3 @@\n two\n-three\n+THREE\n four\n'
    assert apply_patch('one\ntwo\nthree\nfour\nfive\n', patch) == 'one\ntwo\nTHREE\nfour\nfive\n'
    assert apply_patch('one\n', '') == 'one\n'
    assert apply_patch('one\n', HEAD) == 'one\n'


def test_zero_ranges_insert_delete_and_empty():
    assert apply_patch('', HEAD + '@@ -0,0 +1,2 @@\n+a\n+b\n') == 'a\nb\n'
    assert apply_patch('a\nb\n', HEAD + '@@ -1,2 +0,0 @@\n-a\n-b\n') == ''
    patch = HEAD + '@@ -0,0 +1 @@\n+first\n@@ -2,0 +4 @@\n+last\n'
    assert apply_patch('a\nb\n', patch) == 'first\na\nb\nlast\n'
    assert apply_patch('', '') == ''
    assert apply_patch('', HEAD + '@@ -0,0 +0,0 @@\n') == ''


def test_multiple_hunks_use_original_coordinates():
    patch = HEAD + '@@ -1 +1,2 @@\n-a\n+A\n+extra\n@@ -4 +5 @@\n-d\n+D\n'
    assert apply_patch('a\nb\nc\nd\ne\n', patch) == 'A\nextra\nb\nc\nD\ne\n'
    same_gap = HEAD + '@@ -1,0 +2 @@\n+x\n@@ -1,0 +3 @@\n+y\n'
    assert apply_patch('a\nb\n', same_gap) == 'a\nx\ny\nb\n'


def test_blank_content_and_spaces_are_exact():
    patch = HEAD + '@@ -1,2 +1,3 @@\n \n-x  \n+x \n+\n'
    assert apply_patch('\nx  \n', patch) == '\nx \n\n'
    with pytest.raises(ValueError):
        apply_patch('\nx \n', patch)


def test_parser_rejects_malformed_syntax_and_counts():
    assert parse_patch(HEAD) == []
    for patch in ['--- old\n', '--- \n+++ new\n', '--- old\n+++ \n',
                  HEAD + '@@ -0 +1 @@\n-a\n+b\n',
                  HEAD + '@@ -1 +0 @@\n-a\n+b\n',
                  HEAD + '@@ -1 +1 @@ section\n a\n',
                  HEAD + '@@ -1,2 +1 @@\n a\n',
                  HEAD + '@@ -1 +1 @@\n\n',
                  HEAD + '@@ -1 +1 @@\n a\n\\ No newline at end of file\n',
                  HEAD + '@@ -1 +1 @@\n a\n+extra\n']:
        with pytest.raises(ValueError):
            parse_patch(patch)
        with pytest.raises(ValueError):
            apply_patch('a\n', patch)


def test_mismatch_bounds_and_new_coordinates():
    assert apply_patch('a\n', HEAD + '@@ -1 +1 @@\n a\n') == 'a\n'
    for original, body in [('a\n', '@@ -1 +1 @@\n-wrong\n+b\n'),
                           ('a\n', '@@ -1 +1 @@\n wrong\n'),
                           ('a\n', '@@ -3,0 +2 @@\n+x\n'),
                           ('a\n', '@@ -2 +2 @@\n-a\n+b\n'),
                           ('a\n', '@@ -1 +2 @@\n a\n'),
                           ('a\n', '@@ -1 +1,0 @@\n-a\n')]:
        with pytest.raises(ValueError):
            apply_patch(original, HEAD + body)
    with pytest.raises(ValueError):
        apply_patch('unterminated', '')


def test_overlapping_and_reversed_hunks():
    assert apply_patch('a\nb\n', '') == 'a\nb\n'
    for body in ['@@ -1 +1 @@\n a\n@@ -1 +2 @@\n a\n',
                 '@@ -2 +2 @@\n b\n@@ -1 +3 @@\n a\n']:
        with pytest.raises(ValueError):
            apply_patch('a\nb\n', HEAD + body)
