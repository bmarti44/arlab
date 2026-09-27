"""Review fixes: path-safe tags, no traversal in agent.visible, per-question cap enforced after the call, and
interrupted experiments keep their agent-call usage."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from arlab.campaign import Campaign
from arlab.lib.client import BudgetedClient, BudgetExceeded
from arlab.pack import Pack
from arlab.record import load_records, write_json
from conftest import FIXTURE


@pytest.mark.parametrize("tag", ["../x", "/abs", "a/b", "", ".hidden"])
def test_unsafe_tag_rejected(tag, tmp_path):
    with pytest.raises(ValueError):
        Campaign(FIXTURE, tag, runs_root=tmp_path)


def test_visible_traversal_rejected():
    y = yaml.safe_load((FIXTURE / "pack.yaml").read_text())
    y["agent"] = {"visible": ["frozen/run/../eval/evaluate.py"]}
    with pytest.raises(ValueError):
        Pack.model_validate(y)


def test_client_cap_checked_after_call():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = BudgetedClient(f"http://127.0.0.1:{srv.server_port}", "llm", scope_cap=12, max_completion=8)
        assert c.chat("q", [{"role": "user", "content": "hi"}]) == "x"  # 12 tokens: exactly at the cap
        with pytest.raises(BudgetExceeded):
            c.chat("q2", [{"role": "user", "content": "hi"}]) and c.chat("q2", [{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()


def test_interrupted_keeps_agent_calls(tmp_path):
    c = Campaign(FIXTURE, "unit", runs_root=tmp_path)
    rd = c.dir / "runs" / "0001"
    rd.mkdir(parents=True)
    write_json(rd / "calls.json", {"agent_calls": 2, "tokens": {"input": 5, "cached_input": 0, "output": 1}, "timings": {"propose_s": 30.0}})
    c.resume_records()
    r = load_records(c.dir)[0]
    assert r["status"] == "interrupted" and r["agent_calls"] == 2 and r["tokens"]["input"] == 5 and r["timings"]["propose_s"] == 30.0


def test_symlink_out_of_run_output_is_invalid(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "ok.txt").write_text("x")
    (out / "inside").symlink_to(out / "ok.txt")
    (out / "answers.jsonl").symlink_to("/data/private/gold.jsonl")
    escaping = [p for p in out.rglob("*") if p.is_symlink() and not p.resolve().is_relative_to(out.resolve())]
    assert [p.name for p in escaping] == ["answers.jsonl"]
