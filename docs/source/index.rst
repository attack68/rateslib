.. RatesLib documentation master file.

.. image:: _static/rateslib_logo_big.gif
  :alt: Rateslib

.. raw:: html

    <span style="font-size: 4em; color: purple;">BETA v0.0</span>

**This version is pre-release. It may change at any point without notice.**

``Rateslib`` is a state-of-the-art **fixed income library** designed for Python.
Its purpose is to provide advanced, flexible and efficient fixed income analysis
with a high level, well documented API.

Its design objective is to be able to create a self-consistent, arbitrage free
framework for pricing all aspects of fixed income trading, such as spot FX, FX forwards,
single currency securities and derivatives like fixed rate bonds and IRSs, and also
multi-currency derivatives such as FX swaps and cross-currency swaps. Options,
swaptions and inflation are also under consideration for future development.

The techniques and object interation within *rateslib* were inspired by
the requirements of multi-disciplined fixed income teams working, both cooperatively
and independently, within global investment banks.

Highlights
==========

.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOneA">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Curve construction is simple with huge flexibility..
                    </div>
                </div>
            </div>
            <div id="collapseOneA" class="collapse" data-parent="#accordion">
                <div class="card-body">

Multiple interpolation modes are offered by default and the generalised process for curve
solving means very peculiar pricing artefacts can be accurately modelled with the
correct formulations.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseTwoA">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        API is designed from a UI perspective, and is well documented..
                    </div>
                </div>
            </div>
            <div id="collapseTwoA" class="collapse" data-parent="#accordion">
                <div class="card-body">

Although the library uses extensive mathematical processes and models the API has been
carefully designed to provide a workflow that is very simple. In the case of using it
for small scale learning items often few parameters and arguments are required.
For larger series of curves and more complicated object oriented
associations the API signature does not materially change.

The API is also fully documented with examples and advice for best practice.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseThreeA">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Wide range of securities and derivatives are included..
                    </div>
                </div>
            </div>
            <div id="collapseThreeA" class="collapse" data-parent="#accordion">
                <div class="card-body">

The initial beta release of *rateslib* includes all of the standard single currency and
multi-currency instruments. The large array of input parameters for these gives scope
to fully capture the nuances of these products across sectors and geographic regions,
capturing aspects like trading calendars, day count conventions, payment delays, etc.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseFourA">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Limited software dependencies and transparent workflow..
                    </div>
                </div>
            </div>
            <div id="collapseFourA" class="collapse" data-parent="#accordion">
                <div class="card-body">

The dependencies are to **NumPy**, **Pandas**, and **Matplotlib**. *Rateslib* does
not have any dependencies to any automatic
differentiation libraries, such as PyAudi or JAX, preferring initially to use its
own forward mode module.

The test coverage is very high.

.. raw:: html

                </div>
            </div>
        </div>
    </div>
    </div>


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


Get Started
===========

Move on to the next page to :ref:`Get Started<pricing-doc>`

.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    i_get_started.rst
    i_licence.rst
    i_guide.rst
    i_about.rst
    i_api.rst
    i_whatsnew.rst





