For preparing a new release:

On "main":

1) Update the whatsnew with the target release date.
2) Add a new entry to the switcher.json in main:docs/source/static, pushing stable to next version.
3) Change the badges.json file is there is anything to add, e.g. versions.
4) Bump the "version" in pyproject.toml and check the dependencies.
5) Checks should be OK in github actions but perform a local double check.

Checks:
$ coverage run -m pytest
Perform this on development environment as well as specified minimum.
$ pytest -W error
Checking for uncaptured warnings.

6) Commit and push any changes - this will temporarily break readthedocs which will build from push.
7) Create a new release branch, e.g. '0.3.x' and checkout

On "release branch":

1) Update the "release" field in docs/source/conf.py, e.g. to '0.3.x'
2) Delete the switcher in the releases branch since this is taken from main branch
3) Build the INSTRUMENT_SPEC from the loader file and print the dict. Paste to file and set
   DEVELOPMENT to False.
4) Commit and Push the branch.

Build:
$ pip install build twine
$ python -m build
$ twine check dist/*
$ twine upload -r testpypi dist/*
$ twine upload dist/*

check:
$ pip install -i https://test.pypi.org/simple rateslib

In Read-the-Docs admin console:

1) Add a new branch for auto built docs.