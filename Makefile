PHONY: black isort flake8

virtual:

install:
	.env/bin/pip install -e . && env/bin/python post_install.py

black: # Formats code with black
	.env/bin/black --config pyproject.toml ./

isort: isort # Sorts imports using isort
	.env/bin/isort *.py

flake8: .env/bin/flake8 # Lints code using flake8
	.env/bin/flake8 *.py