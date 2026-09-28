"""Synthetic MemoryCode-style bank for stencil-focus (PREPARE only; construction documented in BANK.md).

Deterministic and LLM-free: the paraphrase/distractor text was authored once by Codex in build tooling and frozen
in paraphrases.json; this module only samples from it. Everything is seeded by string seeds (Python's
random.Random(str) is stable across runs and platforms).
"""
from __future__ import annotations

import copy
import random
import re
from collections import defaultdict

# ---------------------------------------------------------------- sizes and MemoryCode template parameters
N_DIALOGUES = {"validation": 300, "holdout": 300, "pilot": 60}
SLICE_INDEX = {"validation": 0, "holdout": 1, "pilot": 2}      # which third of every frozen text list a slice uses
N_SESSIONS = (12, 30)
HISTORY_TOKENS = (6_000, 20_000)          # sizing target (sessions 0..s-1), uniform; measured lengths land ~8-25k
CURRENT_MAX_TOKENS = 1_500                # the current (final) session always fits the window with a 1,024 reminder
CHARS_PER_TOKEN = 4.2                     # sizing estimate only; exact counts are measured and reported
LENGTH_WEIGHT = {"short": 1.0, "medium": 2.0, "long": 3.0}
# generate_dataset.sh values for long dialogues (n >= 20; n < 50)
P_INSTRUCTION_SESSION = (0.5, 0.7)
INSTRUCTIONS_PER_SESSION = (1, 2)
INSTRUCTION_UPDATE_RATE = (0.3, 0.7)
FILLER_TOPIC_RATE = (0.5, 0.7)            # upstream filler_instruction_rate: P(topic filler, i.e. no filler instruction)
FILLER_UPDATE_RATE = (0.5, 0.8)
UPDATABLE = tuple(range(6, 16))           # generate_template.py: instruction_ids_with_updates
# rendering
P_COMPOUND = 0.4          # two convention events of one session in one sentence
P_MIX_ADVICE = 0.15       # a convention or filler instruction shares its sentence with a code-advice VP
P_UPDATE_FRAME = 0.5      # an update event uses an "update" frame
ADVICE_PER_SESSION = (0, 1, 1, 2)
TALK_PER_SESSION = (1, 1, 2, 3)

# ---------------------------------------------------------------- pivot split (IDEA.md: disjoint, stratified)
# Affix pairs across sides (6|11, 7|12, ...) alternating start/end so each side has one updatable pivot per object;
# multi-pivot families spread; the 9 singleton families assigned as recorded here.
SIDE_A = (0, 6, 45, 12, 2, 8, 47, 14, 4, 10, 49,           # class/function/variable/method/attribute/argument
          16, 18, 21, 22, 24,                                # singletons: fn annotation, method try, fn assert,
          25, 27, 29, 31, 35, 39, 42, 34, 38, 43)            #   method docstring, comment | imports | decorators
SIDE_B = (50, 11, 1, 7, 46, 13, 3, 9, 48, 15, 5,
          17, 19, 20, 23,                                    # singletons: method annotation, fn try, method assert, fn docstring
          26, 28, 30, 33, 37, 40, 32, 36, 41, 44)
SIDE_OF_SLICE = {"validation": "A", "holdout": "B"}          # pilot: half A, half B
PIVOTS = {"A": SIDE_A, "B": SIDE_B}

KIND_OF = {0: "case", **{p: "chx" for p in range(1, 6)}, **{p: "prefix" for p in range(6, 11)},
           **{p: "suffix" for p in range(11, 16)}, 16: "annotation", 17: "annotation", 18: "try", 19: "try",
           20: "assert", 21: "assert", 22: "docstring", 23: "docstring", 24: "comment",
           **{p: "import" for p in range(25, 31)}, **{p: "decorator" for p in range(31, 45)},
           **{p: "digit" for p in range(45, 51)}}
SINGULAR = {"function": "function", "variable": "variable", "method": "method", "attribute": "attribute",
            "function argument": "function argument", "class": "class"}
PLURAL = {"function": "functions", "variable": "variables", "method": "methods", "attribute": "attributes",
          "function argument": "function arguments", "class": "classes"}
CASES = ["all uppercase letters", "CamelCase", "snake_case"]   # pivot 0 variants: UPPERCASE, CamelCase, snake_case
TERMINAL = re.compile(r"[.!?]$")


def sentences(text):
    """stencil.focus3.sentences (frozen splitter), duplicated for the bank's self-checks; tests assert equality."""
    return [(m.start(), m.group()) for m in re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", text)]


# ---------------------------------------------------------------- slots of a (pivot, update)
def slots(topic: dict, update: int) -> dict:
    p = int(topic["id"])
    obj, rx = topic["regex"][update]
    base = obj.split(" ")[0] if obj.endswith(("annotation", "try", "assert", "docstring", "decorator")) else obj
    out = {"{obj}": SINGULAR.get(base, base), "{objs}": PLURAL.get(base, base)}
    text = topic["text"][update]
    quoted = re.findall(r"'([^']+)'", text)
    kind = KIND_OF[p]
    if kind in ("prefix", "suffix"):
        out["{affix}"] = quoted[0]
    elif kind == "import":
        out["{module}"] = rx[0]
    elif kind == "decorator":
        out["{decorator}"] = "@" + rx[0]
    elif kind == "case":
        out["{case}"] = CASES[update]
    return out


def fill(template: str, values: dict) -> str:
    for k, v in values.items():
        template = template.replace(k, v)
    if "{" in template or "}" in template:
        raise ValueError(f"unfilled placeholder in {template!r}")
    return template


def frame(fr: str, **vps) -> str:
    s = fr
    for k, v in vps.items():
        s = s.replace("{" + k + "}", v)
    return s[0].upper() + s[1:]


def third(items: list, slice_name: str, key: str) -> list:
    """The slice's disjoint third of a frozen list (seeded permutation per list)."""
    idx = list(range(len(items)))
    random.Random("third:" + key).shuffle(idx)
    k = SLICE_INDEX[slice_name]
    return [items[i] for i in sorted(idx[k::3])]


class Deck:
    """Draw without replacement (reshuffled when empty) so a dialogue rarely repeats a distractor sentence."""

    def __init__(self, rng, items):
        self.rng, self.items, self.left = rng, list(items), []

    def draw(self):
        if not self.left:
            self.left = list(self.items)
            self.rng.shuffle(self.left)
        return self.left.pop()


class Texts:
    """The frozen paraphrase set restricted to one slice (validation | holdout | pilot: disjoint thirds)."""

    def __init__(self, para: dict, slice_name: str):
        self.conv = {k: third(v, slice_name, "conv:" + k) for k, v in para["conventions"].items()}
        # 3 paraphrases per filler variant: paraphrase k belongs to slice k
        self.filler = {int(fid): [v[SLICE_INDEX[slice_name]] for v in variants] for fid, variants in para["fillers"].items()}
        self.advice = third(para["codetalk"]["advice"], slice_name, "advice")
        self.talk = third(para["codetalk"]["talk"], slice_name, "talk")
        self.single = third(para["frames"]["single"], slice_name, "single")
        self.double = third(para["frames"]["double"], slice_name, "double")
        self.update = third(para["frames"]["update"], slice_name, "update")


# ---------------------------------------------------------------- template sampling (port of generate_template.py)
def sample_instructions(rng, topics_side: dict, n_session: int, positions: set, per_session: dict, update_rate: float):
    """Faithful port of vendor/memorycode/code/generate_template.py:sample_instructions with a seeded rng, over one
    side's pivots. One guard added: when no pivot can be introduced or updated, the event is skipped (upstream
    would raise on an empty choice)."""
    topics = {int(t["id"]): copy.deepcopy(t) for t in topics_side}
    by_type = defaultdict(list)
    for pid, t in topics.items():
        by_type[t["regex"][0][0]].append(pid)
    types = list(by_type)
    for t in topics.values():
        t["text_regex"] = [(i, x, r) for i, (x, r) in enumerate(zip(t["text"], t["regex"]))]
    template, kinds, latest, hist_regex, hist_eval = [], [], {}, [], []
    cur_eval, updatable = [], set()
    for si in range(n_session):
        if si not in positions:
            template.append([-1])
            kinds.append([])
            hist_regex.append(list(latest.values()))
            hist_eval.append(list(cur_eval))
            continue
        seen, events, regs, kk = set(), [], {}, []
        for _ in range(per_session[si]):
            sel = set()
            for t in types:
                ids = [p for p in by_type[t] if p in topics]
                if ids:
                    sel.add(rng.choice(ids))
            insert_ids = sorted(sel - updatable - seen)
            update_ids = sorted(updatable - seen)
            if rng.random() < update_rate or not insert_ids:
                if update_ids:
                    pid, kind = rng.choice(update_ids), "instruction-update"
                else:
                    fresh = [p for p in UPDATABLE if p in topics and p not in updatable]
                    if not fresh:
                        if not insert_ids:
                            continue
                        pid, kind = rng.choice(insert_ids), "instruction-add"
                    else:
                        pid, kind = rng.choice(fresh), "instruction-add"
                        updatable.add(pid)
            else:
                pid, kind = rng.choice(insert_ids), "instruction-add"
                if pid in UPDATABLE:
                    updatable.add(pid)
            seen.add(pid)
            tr = topics[pid]["text_regex"]
            k = rng.randint(0, len(tr) - 1) if topics[pid]["shuffle_updates"] else 0
            upd, _text, regex = tr.pop(k)
            events.append([pid, upd])
            kk.append(kind)
            regs[pid] = regex
            if topics[pid]["eval_query"] not in cur_eval:
                cur_eval.append(topics[pid]["eval_query"])
            if not tr:
                topics.pop(pid)
                updatable.discard(pid)
        template.append(events or [-1])
        kinds.append(kk)
        latest.update(regs)
        hist_regex.append(list(latest.values()))
        hist_eval.append(list(cur_eval))
    return template, kinds, hist_regex, hist_eval


def sample_filler_instructions(rng, fillers_instruction: list, n_session: int, no_filler: set, topic_rate: float,
                               update_rate: float):
    """Port of sample_fillers' filler-instruction branch: per session, (filler id, variant, is_update) or None.
    Topic fillers are replaced by the chatter, so a 'topic filler' session gets no filler instruction."""
    fi = {f["id"]: list(range(len(f["text"]))) for f in fillers_instruction}
    updatable, out = set(), []
    for si in range(n_session):
        if si in no_filler or rng.random() < topic_rate:
            out.append(None)
            continue
        update_ids = sorted(updatable)
        if update_ids and rng.random() < update_rate:
            fid, is_update = rng.choice(update_ids), True
        else:
            fid, is_update = rng.choice(sorted(fi)), False
            updatable.add(fid)
        variant = fi[fid].pop()          # upstream pops the last text first
        out.append((fid, variant, is_update))
        if not fi[fid]:
            fi.pop(fid)
            updatable.discard(fid)
    return out


def live_set(template: list, s: int) -> list:
    """stencil.memorycode.live_instructions: latest update per pivot over sessions 0..s, by first introduction."""
    latest, first = {}, {}
    for i in range(s + 1):
        for e in template[i]:
            if isinstance(e, list) and len(e) == 2:
                latest[e[0]] = e[1]
                first.setdefault(e[0], i)
    return sorted(latest.items(), key=lambda pu: (first[pu[0]], pu[0]))


# ---------------------------------------------------------------- chatter
class Chatter:
    """Consumes threads [(role, text), ...] in order; a thread is never reused (its unused rest is discarded)."""

    def __init__(self, threads: list):
        self.threads, self.i, self.used = threads, 0, []

    def next(self):
        if self.i >= len(self.threads):
            raise RuntimeError("chatter pool exhausted")
        t = self.threads[self.i]
        self.i += 1
        self.used.append(t["source"])
        return t["turns"]

    def session(self, target_tokens: float, cap_tokens: float | None = None) -> list:
        """Whole user/assistant exchanges from fresh threads until ~target tokens; each thread's chunk ends on a
        mentor (assistant) turn. With a cap (the current session), exchanges that would overflow it are skipped."""
        turns, est = [], 0.0
        for _ in range(500):
            if turns and est >= target_tokens:
                return turns
            chunk, c_est = [], 0.0
            for role, text in self.next():
                cost = len(text) / CHARS_PER_TOKEN
                if cap_tokens is not None and est + c_est + cost > cap_tokens:
                    break
                chunk.append((role, text))
                c_est += cost
                if est + c_est >= target_tokens and role == "assistant":
                    break
            while chunk and chunk[-1][0] != "assistant":
                c_est -= len(chunk.pop()[1]) / CHARS_PER_TOKEN
            if chunk:
                turns += chunk
                est += c_est
            elif cap_tokens is not None and turns:
                return turns                     # nothing more fits under the cap
        raise RuntimeError("could not fill a session")


# ---------------------------------------------------------------- one dialogue
def insert_units(rng, turns: list, units: list) -> list:
    """Insert each unit sentence into a random mentor turn at a random sentence boundary of that turn."""
    turns = [list(t) for t in turns]
    mentor = [i for i, (r, _) in enumerate(turns) if r == "assistant"]
    for u in units:
        ti = rng.choice(mentor)
        text = turns[ti][1]
        spans = sentences(text)
        cuts = [0] + [a + len(x) for a, x in spans if TERMINAL.search(x)]
        pos = rng.choice(cuts)
        text = (u["text"] + " " + text) if pos == 0 else (text[:pos] + " " + u["text"] + text[pos:])
        turns[ti][1] = text.strip()
    return turns


def build_dialogue(topics: dict, texts: Texts, chatter: Chatter, side: str, seed: str) -> dict:
    rng = random.Random(seed)
    side_topics = [t for t in topics["instructions"] if int(t["id"]) in PIVOTS[side]]
    by_id = {int(t["id"]): t for t in topics["instructions"]}
    for _attempt in range(50):
        n = rng.randint(*N_SESSIONS)
        lengths = rng.choices(list(LENGTH_WEIGHT), k=n)
        k_instr = int(n * rng.uniform(*P_INSTRUCTION_SESSION))
        positions = set(rng.sample(range(n), k_instr))
        only = set(rng.sample(sorted(positions), rng.randint(0, min(n // 5, len(positions)))))
        per = {p: rng.randint(*INSTRUCTIONS_PER_SESSION) for p in positions}
        template, kinds, hist_regex, hist_eval = sample_instructions(rng, side_topics, n, positions, per,
                                                                    rng.uniform(*INSTRUCTION_UPDATE_RATE))
        s = n - 1
        earlier = any(e != [-1] for e in template[:s])
        if hist_regex[s] and earlier:
            break
    else:
        raise RuntimeError(f"{seed}: no usable template")
    fillers = sample_filler_instructions(rng, topics["fillers_instruction"], n, only,
                                         rng.uniform(*FILLER_TOPIC_RATE), rng.uniform(*FILLER_UPDATE_RATE))
    ctx = rng.choice(topics["contexts"])
    live = live_set(template, s)
    query = rng.choice(sorted({by_id[p]["eval_query"] for p, _ in live}))
    advice, talk = Deck(rng, texts.advice), Deck(rng, texts.talk)

    # ---- sessions: chatter sized to a target history length, then the sentences inserted
    total = rng.uniform(*HISTORY_TOKENS)
    w = [LENGTH_WEIGHT[x] for x in lengths[:s]]
    sessions, units_all = [], []
    for i in range(n):
        if i < s:
            turns = chatter.session(total * w[i] / sum(w))
        else:
            turns = chatter.session(min(total * LENGTH_WEIGHT[lengths[i]] / sum(w), 0.8 * CURRENT_MAX_TOKENS),
                                    cap_tokens=CURRENT_MAX_TOKENS)
        units = []
        conv = [e for e in template[i] if e != -1 and isinstance(e, list)]
        conv_kinds = kinds[i]

        def vp_conv(e):
            t = by_id[e[0]]
            return fill(rng.choice(texts.conv[KIND_OF[e[0]]]), slots(t, e[1]))

        def single(vp, is_update):
            pool = texts.update if is_update and rng.random() < P_UPDATE_FRAME else texts.single
            return frame(rng.choice(pool), vp=vp)

        if len(conv) == 2 and rng.random() < P_COMPOUND:
            a, b = rng.sample(conv, 2)
            units.append({"kind": "convention", "conv": [a, b], "update": "instruction-update" in conv_kinds,
                          "text": frame(rng.choice(texts.double), vp1=vp_conv(a), vp2=vp_conv(b))})
        else:
            for e, kind in zip(conv, conv_kinds):
                if rng.random() < P_MIX_ADVICE:
                    pair = [vp_conv(e), advice.draw()]
                    rng.shuffle(pair)
                    txt = frame(rng.choice(texts.double), vp1=pair[0], vp2=pair[1])
                else:
                    txt = single(vp_conv(e), kind == "instruction-update")
                units.append({"kind": "convention", "conv": [e], "update": kind == "instruction-update", "text": txt})
        if fillers[i] is not None:
            fid, var, upd = fillers[i]
            vp = texts.filler[fid][var]
            if rng.random() < P_MIX_ADVICE:
                pair = [vp, advice.draw()]
                rng.shuffle(pair)
                txt = frame(rng.choice(texts.double), vp1=pair[0], vp2=pair[1])
            else:
                txt = single(vp, upd)
            units.append({"kind": "filler", "conv": [], "filler": [fid, var], "update": upd, "text": txt})
        for _ in range(rng.choice(ADVICE_PER_SESSION)):
            units.append({"kind": "advice", "conv": [], "update": False, "text": single(advice.draw(), False)})
        for _ in range(rng.choice(TALK_PER_SESSION)):
            units.append({"kind": "talk", "conv": [], "update": False, "text": talk.draw()})
        rng.shuffle(units)
        turns = insert_units(rng, turns, units)
        names = {"user": ctx["mentee"], "assistant": ctx["mentor"]}
        sessions.append("\n\n".join(f"{names[r]}: {t}" for r, t in turns))
        for u in units:
            units_all.append({**u, "session": i})
    # the latest statement of every live pivot (label-derived oracle, verbatim sentences of sessions 0..s-1)
    last_unit = {}
    for k, u in enumerate(units_all):
        for p, upd in u["conv"]:
            last_unit[(p, upd)] = k
    oracle = sorted({last_unit[(p, u)] for p, u in live if units_all[last_unit[(p, u)]]["session"] < s})
    families = sorted({str(obj) for obj, _ in hist_regex[s]})
    return {"mentor": ctx["mentor"], "mentee": ctx["mentee"], "sessions": sessions, "query": query,
            "template": template, "kinds": kinds, "fillers": [list(f) if f else None for f in fillers],
            "live": [list(x) for x in live], "history_regex": hist_regex[s], "history_eval_query": hist_eval[s],
            "families": families, "units": units_all, "oracle_units": oracle, "n_sessions": n, "side": side}
