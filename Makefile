VENV ?= .venv
PYTHON := $(VENV)/bin/python

.PHONY: check-env test lint format install

check-env:
	@echo "Checking environment..."
	@test -f $(PYTHON) || (echo "ERROR: venv not found at $(VENV). Run 'make install' first." && exit 1)
	@$(PYTHON) -c "import torch; print(f'PyTorch {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
	@echo "✅ Environment OK"

test: check-env
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/

install:
	UV_CACHE_DIR=/srv/zones/Z2-A001/uv-cache uv venv $(VENV) --python 3.12
	UV_CACHE_DIR=/srv/zones/Z2-A001/uv-cache uv pip install --python $(PYTHON) -e '.[dev]'
