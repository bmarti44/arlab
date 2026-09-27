import pytest
from huffman_tree import build_codes
from huffman_codec import encode, decode, compress


def test_known_frequencies_and_bits():
    codes = build_codes("banana")
    assert codes == {"a": "0", "b": "10", "n": "11"}
    assert encode("banana", codes) == "100110110"
    assert decode("100110110", codes) == "banana"
    assert compress("banana") == ("100110110", codes)


def test_ties_include_internal_nodes():
    assert build_codes("dcba") == {"a": "00", "b": "01", "c": "10", "d": "11"}
    assert build_codes("abbccc") == {"a": "00", "b": "01", "c": "1"}
    assert build_codes("ccc bba".replace(" ", "")) == build_codes("abbccc")


def test_empty_and_single_character():
    assert compress("") == ("", {})
    assert encode("", {}) == ""
    assert decode("", {}) == ""
    assert compress("zzzz") == ("0000", {"z": "0"})
    assert decode("000", {"z": "0"}) == "zzz"
    for bits, codes in [("0", {}), ("1", {"z": "0"})]:
        with pytest.raises(ValueError):
            decode(bits, codes)
    with pytest.raises(ValueError):
        encode("x", {})


def test_unicode_whitespace_and_case():
    text = "A a\néé🙂🙂🙂\tA"
    bits, codes = compress(text)
    assert set(codes) == set(text)
    assert decode(bits, codes) == text
    assert bits == encode(text, codes)
    assert build_codes("é🙂") == {"é": "0", "🙂": "1"}


def test_supplied_codebook_and_no_mutation():
    codes = {"a": "11", "b": "0", "c": "10"}
    before = dict(codes)
    assert encode("cab", codes) == "10110"
    assert decode("10110", codes) == "cab"
    assert decode("", codes) == ""
    with pytest.raises(ValueError):
        encode("bad", codes)
    assert codes == before


def test_invalid_codebooks_even_with_empty_input():
    invalid = [{"": "0"}, {"ab": "0"}, {1: "0"}, {"a": ""}, {"a": 1}, {"a": "02"}, {"a": "0", "b": "0"}, {"a": "0", "b": "01"}, {"b": "01", "a": "0"}]
    for codes in invalid:
        with pytest.raises(ValueError):
            encode("", codes)
        with pytest.raises(ValueError):
            decode("", codes)


def test_invalid_and_truncated_bitstreams():
    codes = {"a": "0", "b": "100", "c": "101"}
    assert decode("0100101", codes) == "abc"
    for bits in ("1", "10", "01", "11", "01011", "0x", "0 0", "０"):
        with pytest.raises(ValueError):
            decode(bits, codes)


def test_varied_frequencies_round_trip_and_prefix_property():
    text = "".join(chr(65 + i) * (i + 1) ** 2 for i in range(24))
    bits, codes = compress(text)
    assert decode(bits, codes) == text
    assert set(codes) == set(text)
    assert len(bits) == sum(text.count(char) * len(code) for char, code in codes.items())
    assert len(bits) <= 5 * len(text)
    for char, code in codes.items():
        assert code and set(code) <= {"0", "1"}
        for other, other_code in codes.items():
            if other != char:
                assert not other_code.startswith(code)
