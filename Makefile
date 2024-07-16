PHONY: black isort flake8

virtual:

install:
	pip install -e . && env/bin/python post_install.py

black: # Formats code with black
	black --config pyproject.toml ./

isort: isort # Sorts imports using isort
	isort *.py

flake8: flake8 # Lints code using flake8
	flake8 *.py
