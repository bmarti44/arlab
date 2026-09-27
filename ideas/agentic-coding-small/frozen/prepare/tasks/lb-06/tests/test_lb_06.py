import pytest
from log_parser import LEVELS, parse_line, parse_file
from log_report import summarize, report_file


def test_parse_fields_pipes_and_unicode():
    record = parse_line(" 2024-02-29 01:02:03 | WARN | café | slow \n")
    assert record == {"timestamp": "2024-02-29 01:02:03", "level": "WARN",
                      "message": "café | slow"}
    assert LEVELS == ("INFO", "WARN", "ERROR")


def test_blank_comments_and_invalid_records():
    assert parse_line(" \t\n") is None
    assert parse_line("  # ignored | whatever") is None
    for line in ["bad", "2023-02-29 00:00:00|INFO|x", "2024-1-01 00:00:00|INFO|x",
                 "2024-01-01 24:00:00|INFO|x", "2024-01-01 00:00:60|INFO|x",
                 "2024-01-01 00:00:00|info|x", "2024-01-01 00:00:00|ERROR|  "]:
        with pytest.raises(ValueError):
            parse_line(line)


def test_parse_file_tracks_physical_lines(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("# comment\n\nbroken\n2024-01-02 00:00:00|INFO|ok\n"
                    "bad again\n2024-01-01 00:00:00|ERROR|é\n", encoding="utf-8")
    records, invalid = parse_file(path)
    assert invalid == [3, 5]
    assert [r["message"] for r in records] == ["ok", "é"]
    assert [r["level"] for r in records] == ["INFO", "ERROR"]


def test_summary_generator_and_time_order():
    records = [parse_line("2024-02-02 00:00:00|ERROR|a"),
               parse_line("2024-01-01 00:00:00|INFO|b"),
               parse_line("2024-02-02 00:00:00|ERROR|c")]
    before = [dict(record) for record in records]
    assert summarize(iter(records)) == {
        "total": 3, "counts": {"INFO": 1, "WARN": 0, "ERROR": 2},
        "earliest": "2024-01-01 00:00:00", "latest": "2024-02-02 00:00:00"}
    assert records == before


def test_empty_summary_and_report(tmp_path):
    assert summarize([]) == {"total": 0, "counts": {"INFO": 0, "WARN": 0, "ERROR": 0},
                             "earliest": None, "latest": None}
    path = tmp_path / "empty.log"
    path.write_text("# empty\n\n", encoding="utf-8")
    assert report_file(path) == "TOTAL 0\nINFO 0\nWARN 0\nERROR 0\nFIRST -\nLAST -\nINVALID -\n"


def test_complete_report(tmp_path):
    path = tmp_path / "report.log"
    path.write_text("2024-01-03 12:00:00|WARN|later\nwrong\n"
                    "2024-01-01 00:00:00|INFO|earlier\nwrong\n", encoding="utf-8")
    assert report_file(str(path)) == (
        "TOTAL 2\nINFO 1\nWARN 1\nERROR 0\nFIRST 2024-01-01 00:00:00\n"
        "LAST 2024-01-03 12:00:00\nINVALID 2,4\n")


def test_missing_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_file(tmp_path / "missing.log")
    with pytest.raises(FileNotFoundError):
        report_file(tmp_path / "missing.log")
