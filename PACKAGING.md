# Rateslib Release Schematic

## Overall

There are a number of things to do when packaging a release for *rateslib*.

- Prepare the core code for a release, involving turning off development features and setting file includes.
- Running final tests
- Relinking the Docs URLS with new version tags.
- Building and cross-compiling for different architectures.
- Uploading compiled files to PyPi
- Responding to the Conda auto-PRs or manually constucting new Conda PRs to update conda files.
- Issuing a release in Github.
- Updating the auto-build tags on Read-the-docs.
- Syncing the release with Rateslib-Excel and the documentation website there.
- Notifying users via social media - mainly LinkedIn.
- Setup the library to continue development for a new version.

### Preparing Code, Building and Uploading to PiPy

On the up-to-date *main* branch:

- Update *docs/source/i_whatsnew.rst* with the target release date.
- Add a new entry to *docs/source/_static/switcher.json* pushing to the next intended version.
- Update the *docs/source/_static/badges.json* with the next intended version. Omit conda for now.
- Bump the "version" in pyproject.toml, cargo.toml, and __init__, __version__ 
- Check the dependencies. If anything needs amending consider bumping dependencies and re-starting. 
- Submit a PR and allow github actions to perform checks.

Local checks worth performing:
(Perform this on development environment as well as specified minimum)
$ coverage run -m pytest

(Checking for uncaptured warnings - some of these may be known issues and can be ignored)
$ pytest -W error

- Check cargo.toml::[package][exclude] in case any folders or files need amending (including or excluding). 
- Commit and merge any changes to *main* - this will temporarily break *ReadTheDocs* which will build from push. 
- Create a new release branch, e.g. '1.3.x' and checkout.

On the new release branch *1.3.x*:

- Update the "release" field variable in docs/source/conf.py, e.g. to '0.3.x' 
- Delete the *switcher.json* in this branch to avoid confusion. Switcher is taken from *main*.
- In a console import *rateslib* and print `defaults.spec` and copy the output to the `_spec_loader.py` file as variable INSTRUMENT_SPEC.
- Set DEVELOPMENT in `_spec_loader.py` to *False*.
- Run `ruff format` to restructure the pasted INSTRUMENT_SPEC 
- Commit and Push the branch.

Rust Checks:

- Run `cargo test --lib` to check consistency.
- Ensure the Cargo.toml file has active abi3-py39 default features.
- Endure the [dev-dependencies] and [bench] sections in Cargo.toml are commented out.
- Consider pinning requirements versions (and/or development versions) or make a comment on the
  precise version that was used to build the package. That way future devs can rebuild a package
  exactly. Possibly add a pip freeze file for record.

Build preparations (if not already installed):

$ pip install twine

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

### Updating Read-The-Docs

In Read-the-Docs admin console:
- Activate the new version associated with the created branch, e.g. *1.3.x*.
- *stable* has been removed from RTDs now but force-push the Github *stable* branch to the new release anyway.

### Github Assets Release

- Add a new release on the Github page, which packages release assets.
- Update the Repo *Social Preview" PNG with the new badges.

### Syncing with Rateslib-Excel

- Run the separate *rateslib-excel* checks once a new *rateslib* version is available and update that package separately.

There should always be a /latest/ release named "dev" for rateslib and rateslib-excel.
Each "dev" release can point to each other's hyperlinks on the /latest/ directory.
For rateslib-excel the files it points towards at rateslib should be for a specific version. This is usually 
configurable in the conf intersphinx section.
The video tutorials in the rateslib-excel docs that are pointed to from rateslib can be the /latest/ files.

### Updating Conda packages

- Allow conda to detect PiPy releases and run its auto-PRs (usually within 24 hours).
- Merge these PRs to *rateslib-feedstock* if all tests successful. Otherwise panic and cry for help from conda build community.

### Resetting the Main branch for continued development

To Make a New Release:

- Edit Cargo.toml and update the version.
- Update the version in pyproject.toml.
- Update the version in rateslib/__init__.py
- Update the version test in test_default.py
- Add a new release table to the whats new doc page.
- Docs conf.py should record the release name version as "dev".
- spec loader DEVELOPMENT flag should be True
