.PHONY: install remove start zip

VENV_DIR := .venv
PYTHON := /opt/anaconda3/envs/py311/bin/python 
ARCHIVE := archive.zip


install:
	@echo "🔧 Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created."

	@echo "📦 Installing Poetry..."
	@. $(VENV_DIR)/bin/activate && pip install poetry
	@echo "✅ Poetry installed."

	@echo "📚 Installing dependencies with Poetry..."
	@. $(VENV_DIR)/bin/activate && poetry install --no-root
	@echo "✅ Dependencies installed."

	@echo "🧹 Setting up pre-commit hooks..."
	@. $(VENV_DIR)/bin/activate && pre-commit install
	@echo "✅ Pre-commit hooks installed."

remove:
	@echo "🗑️  Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "✅ Virtual environment removed."

start:
	@echo "🚀 Starting application..."
	$(VENV_DIR)/bin/python $(PWD)/app/main.py
	@echo "✅ Application exited."

zip:
	@echo "📦 Zipping all folders and files (excluding $(VENV_DIR) folder)..."
	@zip -r $(ARCHIVE) . -x "$(VENV_DIR)/*" -x ".mypy_cache/*" -x "$(ARCHIVE)"
	@echo "✅ Archive '$(ARCHIVE)' created."