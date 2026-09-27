import pytest
from morse import CODES, MorseCodec, normalize_text, split_morse


def test_encode_normalizes_words():
    codec = MorseCodec()
    assert codec.encode("  sOs\tHelp\n2 ") == "... --- ... / .... . .-.. .--. / ..---"
    assert normalize_text("a\u2003b 09") == ["A", "B", "09"]


def test_decode_and_word_boundaries():
    codec = MorseCodec()
    assert codec.decode("... --- ... / .---- ..---") == "SOS 12"
    assert split_morse(".- -... / -.-.") == [[".-", "-..."], ["-.-."]]


def test_full_alphabet_round_trip():
    codec = MorseCodec()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    expected = ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.. ----- .---- ..--- ...-- ....- ..... -.... --... ---.. ----."
    assert codec.encode(alphabet.lower()) == expected
    assert codec.decode(expected) == alphabet
    assert set(CODES) == set(alphabet)


def test_empty_inputs():
    codec = MorseCodec()
    assert codec.encode(" \t\n") == codec.decode("") == ""
    assert normalize_text("") == split_morse("") == []


def test_invalid_plaintext():
    codec = MorseCodec()
    for value in ["Hi!", "café", "ß", "１２", "a/b", "a_b"]:
        with pytest.raises(ValueError):
            normalize_text(value)
        with pytest.raises(ValueError):
            codec.encode(value)


def test_noncanonical_wire_format():
    codec = MorseCodec()
    for value in [" ", " .-", ".- ", ".-  -...", ".-/-...", ".- / / -...", " / .-", ".- / ", ".-\t-...", ".-\n", ".x"]:
        with pytest.raises(ValueError):
            split_morse(value)
        with pytest.raises(ValueError):
            codec.decode(value)


def test_unknown_code_is_not_a_format_error():
    assert split_morse("...... / .-") == [["......"], [".-"]]
    with pytest.raises(ValueError):
        MorseCodec().decode("...... / .-")
