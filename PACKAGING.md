For preparing a new release:

On "main":

1) Update the whatsnew with the target release date.
2) Add a new entry to the switcher.json in main:docs/source/static, pushing stable to next version.
3) Bump the "version" in pyproject.toml and check the dependencies.
4) Checks should be OK in github actions but perform a local double check.

Checks:
$ coverage run -m pytest
Perform this on development environment as well as specified minimum.
$ pytest -W error
Checking for uncaptured warnings.

5) Commit and push any changes - this will temporarily break readthedocs which will build from push.


Branch to version: x.x.x, e.g. 0.2.x
Update "release" in docs/source/conf.py, e.g. to 0.2.x
Update the switcher.json in main:docs/source/static
Delete the switcher in the releases branch since this is taken from main.
Once reverted back to main, switch "release" back to dev. 



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