
Branch to version: x.x.x, e.g. 0.2.x
Update "release" in docs/source/conf.py, e.g. to 0.2.x
Update the switcher.json in main:docs/source/static
Update pyproject.toml with new version.
Delete the switcher in the releases branch since this is taken from main.
Add release date to whatsnew file.
Once reverted back to main, switch "release" back to dev. 

Checks:
$ coverage run -m pytest
Perform this on development environment as well as specified minimum.

Build:
$ pip install build twine
$ python -m build
$ twine check dist/*
$ twine upload -r testpypi dist/*
$ twine upload dist/*

check:
$ pip install -i https://test.pypi.org/simple rateslib

docs:
Goto read-the-docs.io and add a new branch.