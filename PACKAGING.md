For preparing a new release:

On "main":

1) Update the whatsnew with the target release date.
2) Add a new entry to the switcher.json in main:docs/source/static, pushing stable to next version.
3) Change the badges.json file is there is anything to add, e.g. versions.
4) Bump the "version" in pyproject.toml, cargo.toml, and __init__ __version__ and check the dependencies.
5) Checks should be OK in github actions but perform a local double check.

Checks:
$ coverage run -m pytest
Perform this on development environment as well as specified minimum.
$ pytest -W error
Checking for uncaptured warnings.

6) Check cargo.toml excludes in case any folders or files need amending.
7) Commit and push any changes - this will temporarily break readthedocs which will build from push.
8) Create a new release branch, e.g. '0.3.x' and checkout

On "release branch":

1) Update the "release" field in docs/source/conf.py, e.g. to '0.3.x'
2) Delete the switcher in the releases branch since this is taken from main branch
3) Build the INSTRUMENT_SPEC from the loader file and print the dict. Paste to file and set
   DEVELOPMENT to False.

4) Commit and Push the branch.
5) Run `cargo test --lib` to check consistency
6) Ensure the Cargo.toml file has active abi3-py39 features.
7) Comment out the benchmark code and dev section code (otherwise source distribution will not run)

pip install twine (if necessary)

Rust Extension Build:
Build:  https://doc.rust-lang.org/rustc/platform-support.html
$ maturin build --release
$ maturin build --release --sdist
$ maturin build --release --target aarch64-apple-darwin
$ maturin build --release --target x86_64-apple-darwin

$ twine check target/wheels/*
$ twine upload -r testpypi target/wheels/*

Test with virtual environment to install from wheels and from source.
($ pip install pandas numpy matplotlib) to avoid sources not available on testpypi
$ pip install -i https://test.pypi.org/simple rateslib

$ twine upload target/wheels/*  [use __token__ as username and token is in env file]


In Read-the-Docs admin console:

1) Add a new branch for auto built docs.
2) Checkout the "stable" tagging branch and update to the new version and force push. 

In GITHUB:

1) Add a new release.
2) Update the Repo PNG with the new badges.


To Make a New Release:

1) Edit Cargo.toml and update the version.
2) Update the version in pyproject.toml.
3) Update the version in rateslib/__init__.py
4) Update the version test in test_default.py
5) Add a new release table to the whats new doc page.
6) Docs conf.py should record the release name version as "dev".