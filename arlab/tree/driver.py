"""Runs inside the policy sandbox: loads /p/policy.py and answers one JSON line per request."""
import importlib.util
import json
import sys
import traceback

out = sys.stdout
sys.stdout = sys.stderr  # print() inside a policy must not corrupt the protocol
try:
    spec = importlib.util.spec_from_file_location("policy", "/p/policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    err = None
except Exception:
    err = traceback.format_exc()
for line in sys.stdin:
    try:
        if err:
            raise RuntimeError("import failed:\n" + err)
        q = json.loads(line)
        sel = mod.select(q["nodes"], q["budget_left"], q["W"])
        res = {"select": sel if isinstance(sel, list) else {"not a list": repr(sel)[:200]}}
    except Exception:
        res = {"error": traceback.format_exc()}
    out.write(json.dumps(res) + "\n")
    out.flush()
