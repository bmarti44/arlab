import pytest
from text_stats import TextStats, syllables, words


def test_ascii_word_rules():
    assert words("Don't STOP--now42café; rock’n’roll 'x'") == ["don't", 'stop', 'now', 'caf', 'rock', 'n', 'roll', 'x']
    assert words("a'b'c") == ["a'b'c"]


def test_syllable_heuristic():
    assert [syllables(w) for w in ['make', 'table', 'sky', 'queue', 'rhythm', 'AEON', 'the', 'brr', "we're"]] == [1, 2, 1, 1, 1, 1, 1, 1, 1]
    for word in ('', 'two words', 'abc2', "'abc", 'café'):
        with pytest.raises(ValueError):
            syllables(word)


def test_sentence_chunks_and_score():
    text = '... Cat sat!?! A table. Last'
    assert TextStats(text).summary() == {
        'words': 5, 'sentences': 3, 'syllables': 6,
        'characters': len(text), 'reading_ease': 103.62}


def test_empty_and_punctuation_only():
    s = TextStats()
    assert s.summary() == {'words': 0, 'sentences': 0, 'syllables': 0, 'characters': 0, 'reading_ease': 0.0}
    s.replace('12...!? 99')
    assert s.summary() == {'words': 0, 'sentences': 0, 'syllables': 0, 'characters': 10, 'reading_ease': 0.0}
    assert s.most_common() == []


def test_editing_and_fresh_reports():
    s = TextStats('ca')
    s.summary()['words'] = 100
    s.append('t. Dog')
    assert s.most_common() == [('cat', 1), ('dog', 1)]
    assert s.summary()['sentences'] == 2
    s.replace('table')
    assert s.summary() == {'words': 1, 'sentences': 1, 'syllables': 2, 'characters': 5, 'reading_ease': 36.62}


def test_frequency_sort_limits_and_validation():
    s = TextStats('B a b A c d e f')
    assert s.most_common() == [('a', 2), ('b', 2), ('c', 1), ('d', 1), ('e', 1)]
    assert s.most_common(2) == [('a', 2), ('b', 2)]
    assert s.most_common(0) == []
    for limit in (-1, True, 1.5):
        with pytest.raises(ValueError):
            s.most_common(limit)


def test_unterminated_sentences_and_unclamped_scores():
    s = TextStats('Cat\ndog')
    assert s.summary()['sentences'] == 1
    assert s.summary()['reading_ease'] == 120.21
    long_word = TextStats('abababababababa')
    assert long_word.summary()['reading_ease'] == -470.98
