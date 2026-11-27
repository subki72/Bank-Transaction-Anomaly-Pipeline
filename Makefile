.PHONY: install lint format test health demo run docker-build clean

install:
	python -m pip install -r requirements.txt

lint:
	python -m ruff check src tests main.py
	python -m ruff format --check src tests main.py
	python -m flake8 --max-line-length=120 src tests main.py

format:
	python -m ruff format src tests main.py

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75

health:
	python main.py --health

demo:
	python main.py --demo --rows 400 --seed 42

run:
	python main.py --serve --host 0.0.0.0 --port 8000

docker-build:
	docker build -t bank-anomaly-pipeline:latest .

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in (pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('htmlcov'))]"
	python -c "import pathlib; [p.unlink(missing_ok=True) for p in (pathlib.Path('.coverage'), pathlib.Path('coverage.xml'))]"
