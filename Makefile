.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
JAX_INSTALLED=$(shell python -c "import jax")

.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '.ipynb_checkpoints' -exec rm -rf {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@find ./ -name '*.asv' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: jaxgpu
jaxgpu:           ## Installs jax for *nix systems with CUDA
	@echo "Installing jax..."
	@$(ENV_PREFIX)pip install --upgrade pip
	@$(ENV_PREFIX)pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


.PHONY: lint
lint:             ## Runs isort and mypy.
	@echo "Running isort ..."
	$(ENV_PREFIX)isort jwave/
	@echo "Running flake8 ..."
	$(ENV_PREFIX)flake8 jwave/  --count --select=E9,F63,F7,F82 --show-source --statistics
	$(ENV_PREFIX)flake8 jwave/ --count --ignore=E111 --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Running mypy ..."
	$(ENV_PREFIX)mypy --allow-redefinition --config-file=pyproject.toml jwave/*.py

.PHONY: virtualenv
virtualenv:       ## Create a virtual environment. Checks that python > 3.8
	@echo "creating virtual environment ..."
	@python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8 or higher is required'" || exit 1
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo "Installing jax with GPU support"
	@make jaxgpu
	@echo "Installing training requirements"
	@./.venv/bin/pip install -r requirements-train.txt
	@echo "Instaling FNO"
	@./.venv/bin/pip install -e .
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"

.PHONY: test
test:             ## Run tests with pytest (discover mode)
	$(ENV_PREFIX)pytest


.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | entr $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/


# This makefile has been adapted from rochacbruno/python-project-template
