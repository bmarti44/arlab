# Acceptance targets: the exit code is the verdict (PLAN §0).
PY := .venv/bin/python
ARLAB := .venv/bin/arlab

.PHONY: test accept-M0 accept-M0-cpu accept-M1 accept-M2 accept-M3 accept-M4 accept-M5-memory-longmemeval accept-M5-agentic-coding-small accept-M6

test:
	$(PY) -m pytest -q tests -m "not docker and not spark"

accept-M0:
	scripts/accept_m0.sh

accept-M0-cpu:
	scripts/accept_m0.sh --cpu-only

accept-M1:
	$(PY) -m pytest -q tests -m "not spark"

accept-M2:
	scripts/accept_m2.sh

accept-M3:
	$(PY) scripts/accept_campaign.py m3

accept-M4:
	$(PY) scripts/accept_campaign.py m4

accept-M5-memory-longmemeval:
	$(PY) scripts/accept_campaign.py m5 memory-longmemeval

accept-M5-agentic-coding-small:
	$(PY) scripts/accept_campaign.py m5 agentic-coding-small

accept-M6:
	$(PY) scripts/accept_campaign.py m6
