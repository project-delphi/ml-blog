# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
# ---

# %%
"""Setup package for local live installation."""
from __future__ import annotations

# %%
from setuptools import setup

# %%
with open("README.md") as f:
    long_description = f.read()

# %%
setup(
    name="my_package",
    version="0.1.0",
    packages=["my_package"],
    install_requires=["dependency1", "dependency2"],
    author="Your Name",
    description="A short description of your package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/my_package",
)
