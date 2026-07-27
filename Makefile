# setup blog virtual environment
venv:
	uv venv .venv
# install dependencies for blog
install:
	uv pip install --python .venv/bin/python .
# register the shared blog-base Jupyter kernel over .venv
kernel:
	.venv/bin/python -m ipykernel install --user --name blog-base
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
	@echo "kernel - register the blog-base Jupyter kernel"
	@echo "preview - quarto preview"
	@echo "quatro - quarto render"
	@echo "help - help"
