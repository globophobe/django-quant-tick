VENV := .venv/bin
PYTHON := $(VENV)/python
RUFF := $(VENV)/ruff

.PHONY: check lint format test

check: lint test

lint:
	$(RUFF) check .

format:
	$(RUFF) check . --fix

test:
	cd demo && ../$(PYTHON) manage.py test quant_tick --verbosity 1
