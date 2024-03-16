# setup blog virtual environment
venv:
	python3 -m venv .venv; \
	source .venv/bin/activate;
# install dependencies for blog
install:
	pip install --upgrade pip; \
	pip install .
# quarto preview
preview:
	quarto preview .
# quarto render
quatro:
	quarto render .
# help
help:
	@echo "venv - setup blog virtual environment"
	@echo "install - install dependencies for blog"
	@echo "preview - quarto preview"
	@echo "quatro - quarto render"
	@echo "help - help"
