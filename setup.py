from setuptools import find_packages, setup

PACKAGE_NAME = "python-scaffold-development"
SHORT_DESCRIPTION = "A short description of your package"
URL_STRING = "https://github.com/project-delphi/python-scaffold-development"

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements/requirements_run.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    author="Ravi Kalia",
    author_email="ravkalia@gmail.com",
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL_STRING,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
