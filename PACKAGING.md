
Branch to version: x.x.x, e.g. 0.2.x
Update "release" in docs/source/conf.py, e.g. to 0.2.x
Update the switcher.json in main:docs/source/static
Update pyproject.toml with new version.

Checks:
$ coverage run -m pytest
Perform this on development environment as well as specified minimum.

Build:
$ pip install build twine
$ python -m build
$ twine check dist/*
$ twine upload -r testpypi dist/*

check:
$ pip install -i https://test.pypi.org/simple rateslib