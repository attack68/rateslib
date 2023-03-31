.. RatesLib documentation master file.

.. image:: _static/rateslib_logo_big.gif
  :alt: Rateslib

.. raw:: html

    <span style="font-size: 4em; color: purple;">BETA v0.0</span>

**This version is pre-release. It may change at any point without notice.**

``Rateslib`` is a state-of-the-art fixed income library designed for Python.
Its purpose is to provide advanced, flexible and efficient fixed income analysis
with a high level, well documented API.

The techniques developed for ``rateslib`` were inspired by the requirements of
multi-disciplined fixed income teams working, both cooperatively and independently,
within global investment banks.

Licence
=======

This library is released under a **Creative Commons Attribution, Non-Commercial,
No-Derivatives 4.0 International Licence**.

.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        <span style="color: darkgreen;">Activities allowed</span>&nbsp; under this licence..
                    </div>
                </div>
            </div>
            <div id="collapseOne" class="collapse" data-parent="#accordion">
                <div class="card-body">

- Download and use the code privately in a non-profit function, such as;

  - a learning environment,
  - or for academic uses,

- Modify the code for **private, non-commercial use** only.
- Share or redistribute the code, but only in its **entirety without modification**, and **with attribution**,
  and such that **end use is non-commercial** and in a non-profit function.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseTwo">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        <span style="color: firebrick;">Activities prohibited</span>&nbsp; under this licence..
                    </div>
                </div>
            </div>
            <div id="collapseTwo" class="collapse" data-parent="#accordion">
                <div class="card-body">

- Use the code for any form of commercial or profit based activity. You cannot use it
  inside a bank, a fund, or any form of investment firm where its use is part of the
  operation, or decision making process, of that entity.
- Copy or modify the code and use it in any derivative product or commercial activity,
  such as in a trading application, API or other form of publication.
- Share or redistribute any **sub-part** of the code, or **modified** code, even with attribution.
- Share or redistribute the code in its entirety **without attribution**.
- Include this code, or package, as a **dependency** of any other package.
- Use this code as a benchmark or as a validator for developing you own code
  which will be used in a commercial capacity.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseThree">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        <span style="color: darkorange;">Other licence options available...</span>
                    </div>
                </div>
            </div>
            <div id="collapseThree" class="collapse" data-parent="#accordion">
                <div class="card-body">

If you would like to discuss licensing, under very reasonable terms, please contact the
author at **rateslib@gmail.com**.

.. raw:: html

                </div>
            </div>
        </div>
    </div>
    </div>

Highlights
==========

Constructing curves is easy:

- Specify them directly with nodes.
- Import them pre-constructed from a server or database via JSON.
- Solve their node values with through an optimisation algorithm.

API designed with a UI focus

Fully documented, with extended documentation in book form.

Code test coverage of 100%.

Limited software dependencies:

- NumPy
- Pandas
- Scipy
- Matplotlib

Automatic differentiation library is internal requiring no external libraries.


Contents
========

.. toctree::
    :maxdepth: 0
    :titlesonly:

    i_licence.rst
    i_about.rst
    i_get_started.rst
    i_guide.rst

:ref:`API Reference<api-doc>`

.. toctree::
    :hidden:

    i_api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
