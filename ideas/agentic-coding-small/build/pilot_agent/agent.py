"""Editable surface: a minimal bash-only coding agent (mini-swe-agent style).

solve(task, llm, tools) works in the task's directory until it finishes or runs out of steps/tokens/time.
task  = {id, title, instructions, files}
llm   = llm.chat(messages, max_tokens=None, stop=None) -> str  (temperature 0; llm.tokens_left)
tools = tools.read(path) / write(path, content) / edit(path, old, new) / run(cmd, timeout=60) / finish()
"""
import json
import re
import time

SYSTEM = """You are a careful software engineer working in a Linux shell, in the task's working directory.
Each reply: a short THOUGHT, then exactly ONE bash code block with the command(s) to run, like

THOUGHT: I will look at the files.
```bash
ls -la && cat main.py
```

You will see the command output. Create or edit files with shell tools (cat <<'EOF' > file, sed, python3).
Test your work (write quick checks, run python3 / pytest). When the task is fully done, reply with a block
containing only: echo TASK_DONE"""

KEEP_TURNS = 12


def solve(task, llm, tools):
    first = (f"# Task: {task['title']}\n\n{task['instructions']}\n\n"
             f"Files in the working directory: {', '.join(task['files']) or '(none)'}")
    history = []
    while True:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}] + history[-2 * KEEP_TURNS:]
        t0 = time.time()
        reply = llm.chat(msgs, max_tokens=2048)
        with open(f"/out/log-{task['id']}.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time() - t0, 1), "tokens_left": llm.tokens_left, "reply": reply}) + "\n")
        blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", reply, re.S)
        history.append({"role": "assistant", "content": reply})
        if len(blocks) != 1:
            history.append({"role": "user", "content": "Format error: reply with a THOUGHT and exactly ONE ```bash block."})
            continue
        cmd = blocks[0].strip()
        if cmd == "echo TASK_DONE":
            tools.finish()
            return
        out = tools.run(cmd)
        with open(f"/out/log-{task['id']}.jsonl", "a") as f:
            f.write(json.dumps({"cmd": cmd, "out": out[-1500:]}) + "\n")
        history.append({"role": "user", "content": f"<output>\n{out[-3000:]}\n</output>"})
