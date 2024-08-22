.. _developer-doc:

.. role:: red

**************
Developers
**************

*Rateslib* is a project designed for Python built with Python and Rust.
To actively develop it and submit pull requests (PRs) to the repo you will need
to set up a development environment.

**1) Get the files**

Clone the repository and navigate to the directory.

.. code-block::

   />$ git clone https://github.com/attack68/rateslib.git
   />$ cd rateslib

**2) Setup Python and Rust**

It is recommended to **install Python 3.12 and Rust 1.80**.

Create and activate a virtual Python environment, as below
(or the equivalent for Windows).

.. code-block::

   rateslib/>$ python3 -m venv venv
   rateslib/>$ source ./venv/bin/activate

**3) Install development requirements**

Install the package dependencies for development and
then install *rateslib* locally in editable mode.

.. code-block::

   (venv) rateslib/>$ pip install -r requirement.txt
   (venv) rateslib/>$ pip install -e .

**4) Test your environment**

After installing everything test that everything is successful.

.. code-block::

   (venv) rateslib/>$ pytest

**5) Making changes**

You can now edit and make code changes and submit them for addition to
the package. The continuous integration (CI) checks on github perform 7 tasks:

- **Rust Checks**

  If you do not make changes to the *Rust* section of the code these will remain
  unaffected. If you do it will run:

  - ``cargo fmt --check``: to ensure that rust code is properly formatted. If you
    need to make auto-formatting changes then run the following before committing:

    .. code-block::

       (venv) rateslib/>$ cargo fmt

  - ``cargo test --lib``: to ensure that any introduced code does not impact the
    pre-existing unit tests for the existing codebase. To check this before
    committing run:

    .. code-block::

      (venv) rateslib/>$ cargo test --lib

  - ``cargo test --doc``: to ensure that any introduced code does not impact the
    pre-existing tests in the rust documentation. To check this before
    committing run:

    .. code-block::

      (venv) rateslib/>$ cargo test --doc

  If you made changes to the *Rust* section then you must rebuild the PyO3 extension
  module before testing it works in Python. To do this run:

  .. code-block::

     (venv) rateslib/>$ maturin develop --release

- **Python Checks**

  If you make changes to the Python section of the code it will run:

  - ``ruff check``: to ensure that no styling or linting errors have been introduced.
    You may be able to auto-fix any of these before committing by using:

    .. code-block::

       (venv) rateslib/>$ ruff check --fix

  - ``ruff format``: to ensure that the Python code is correctly formatted. If the CI
    reports errors you can auto-format your code by locally running:

    .. code-block::

       (venv) rateslib/>$ ruff format

  - ``coverage run -m pytest``: this runs all the pytests and measures the coverage
    report of the codebase which should remain > 96%. You can this locally also
    before committing by using:

    .. code-block::

       (venv) rateslib/>$ coverage run -m pytest

- **Building Documentation**

  The documentation is built with *Sphinx* and its extensions. To locally build
  the documentation to debug or visualise any changes before submission, run:

  .. code-block::

   (venv) rateslib/>$ cd docs
   (venv)     docs/>$ make clean
   (venv)     docs/>$ make html

  In order to build docs without error you will also need to **install pandoc**.

  In order to generate inheritance diagrams you will need to **install graphviz**.
