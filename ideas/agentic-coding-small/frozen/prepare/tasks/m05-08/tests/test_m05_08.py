import pytest
from contact_rules import normalize_email, normalize_phone, normalize_contact
from address_book import AddressBook


def test_normalization_and_stable_phone_dedup():
    assert normalize_email(" A@EXAMPLE.Org ") == "a@example.org"
    assert normalize_phone("+1 (202)-555.0100") == "+12025550100"
    assert normalize_contact({"email": " X@Y ", "name": " Ada ", "phones": ["12-3", "123", "+123", "4"], "extra": 9}) == {"name": "Ada", "email": "x@y", "phones": ["123", "+123", "4"]}
    assert normalize_contact({"email": "x@y"}) == {"name": "", "email": "x@y", "phones": []}


def test_invalid_normalization():
    for email in ["", "a", "@b", "a@", "a@@b", "a b@c", "a@b\tc"]:
        with pytest.raises(ValueError):
            normalize_email(email)
    for phone in ["", "+", "12x", "1+2", "++12", "１２", "1\t2", "1" * 16]:
        with pytest.raises(ValueError):
            normalize_phone(phone)
    with pytest.raises(ValueError):
        normalize_contact({"name": "No email"})


def test_upsert_name_and_phone_merge_rules():
    book = AddressBook()
    book.upsert({"email": "a@b", "name": "Ada", "phones": ["123", "456"]})
    assert book.upsert({"email": " A@B ", "name": " ", "phones": ["4-56", "789"]}) == {"name": "Ada", "email": "a@b", "phones": ["123", "456", "789"]}
    assert book.upsert({"email": "a@b", "name": " Grace "})["name"] == "Grace"
    assert len(book.contacts()) == 1


def test_batch_distinct_count_and_sequential_merges():
    book = AddressBook()
    book.upsert({"email": "z@x", "name": "Old"})
    records = [{"email": "Z@X", "name": "New"}, {"email": "a@x", "phones": ["1"]}, {"email": "a@X", "name": "Last", "phones": ["2", "1"]}, {"email": "z@x", "name": ""}]
    assert book.merge(record for record in records) == 2
    assert book.contacts() == [{"name": "Last", "email": "a@x", "phones": ["1", "2"]}, {"name": "New", "email": "z@x", "phones": []}]
    assert book.merge([]) == 0


def test_failed_batch_and_upsert_are_atomic():
    book = AddressBook()
    book.upsert({"email": "a@b", "name": "Original"})
    before = book.contacts()
    with pytest.raises(ValueError):
        book.merge([{"email": "a@b", "name": "Changed"}, {"email": "new@b"}, {"email": "bad"}])
    assert book.contacts() == before
    with pytest.raises(ValueError):
        book.upsert({"email": "a@b", "name": "Changed", "phones": ["1", "bad"]})
    assert book.contacts() == before


def test_lookup_remove_and_invalid_keys():
    book = AddressBook()
    assert book.get("absent@x") is None
    assert book.remove("absent@x") is False
    book.upsert({"email": "a@b"})
    assert book.get(" A@B ")["email"] == "a@b"
    assert book.remove(" A@B ") is True
    assert book.contacts() == []
    for method in [book.get, book.remove]:
        with pytest.raises(ValueError):
            method("invalid")


def test_inputs_and_upsert_result_are_detached():
    record = {"email": "a@b", "phones": ["123"]}
    book = AddressBook()
    result = book.upsert(record)
    record["phones"].append("456")
    result["phones"].clear()
    result["name"] = "Changed"
    assert book.get("a@b") == {"name": "", "email": "a@b", "phones": ["123"]}


def test_listing_and_lookup_are_detached():
    book = AddressBook()
    book.merge([{"email": "b@x", "phones": ["1"]}, {"email": "a@x"}])
    listing = book.contacts()
    listing[1]["phones"].append("2")
    listing.reverse()
    lookup = book.get("b@x")
    lookup["phones"].clear()
    assert [record["email"] for record in book.contacts()] == ["a@x", "b@x"]
    assert book.get("b@x")["phones"] == ["1"]
