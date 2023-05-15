
Branch to version: x.x.x
Update "release" in docs/source/conf.py
Update the switcher.json in main:docs/source/static

Build:
$ coverage run -m pytest
$ pip install build twine
$ python -m build
$ twine check dist/*
$ twine upload -r testpypi dist/*

check:
$ pip install -i https://test.pypi.org/simple rateslib