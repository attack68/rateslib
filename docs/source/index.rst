.. RatesLib documentation master file.

.. ipython:: python
   :suppress:

   from rateslib import *
   from pandas import DataFrame, Series


.. raw:: html

   <div style="text-align: center; padding: 2em 0em 3em;">
       <img alt="rateslib" src="_static/rateslib_logo_big2.png">
   </div>

   <div style="text-align: center">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.python&label=Python&color=blue" alt="Python">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.pypi&label=PyPi&color=blue" alt="PyPi">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.conda&label=Conda&color=blue" alt="Conda">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.licence&label=Licence&color=orange" alt="Licence">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.status&label=Status&color=green" alt="Status">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.coverage&label=Coverage&color=green" alt="Coverage">
      <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Frateslib.com%2Fpy%2F%2Fen%2Flatest%2F_static%2Fbadges.json&query=%24.style&label=Code%20Style&color=black" alt="Code Style">
   </div>

   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <div style="text-align: center; padding: 5px 0em 2.5em 0em;">
   <a class="github-button" href="https://github.com/attack68/rateslib" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star attack68/rateslib on GitHub">Star</a>
   <a class="github-button" href="https://github.com/attack68/rateslib/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="true" aria-label="Issue attack68/rateslib on GitHub">Issue</a>
   </div>

*Rateslib* is a state-of-the-art **fixed income library** designed for Python.
Its purpose is to provide advanced, flexible and efficient fixed income analysis
with a high level, well documented API.

Licence
=======

*Rateslib* is released under an **Amended Creative Commons Attribution, Non-Commercial,
No-Derivatives 4.0 International Licence**. For more details see :ref:`here<licence-doc>`.

.. raw:: html

   <span style="font-size:1.5em;font-weight:bold;">Commercial use:</span> <span style="color:red; font-size: 1.15em;">Not licensed for any use (funds, banks, etc,).</span>
   <a class="purchase-btn" href="i_purchase.html">Purchase Licence Extension</a><br>
   <span style="font-size:1.5em;font-weight:bold;">Academic and at-home educational use:</span> <span style="color:green; font-size: 1.5em; font-weight:bold;">FREE to use and modify.</span>


.. raw:: html

   <div class="clear" style="text-align: center; padding: 1em 0em 1em;"></div>

Highlights
==========

*Other interface bindings*
----------------------------

Extension websites provide the documentation for using *rateslib* in other ways than directly
with Python.

.. raw:: html

   <div class="clear" style="text-align: center; padding: 1em 0em 1em;"></div>

   <div class="flex-container-line">
     <div class="flex-item">
       <a href="https://rateslib.com/excel/latest/"><img src="_static/rateslib_excel_logo_small.png" alt="Rateslib-Excel" width="170" height="37"></a>
     </div>
     <div class="flex-item-right">
       <a href="https://rateslib.com/excel/latest/">rateslib-excel</a> provides Excel bindings for the Python library, to users
       with a <a href="https://rateslib.com/py/en/latest/i_licence.html" target="_blank">commercial licence extension</a>.
     </div>
   </div>
   <div class="flex-container-line">
     <div class="flex-item">
       <a href="https://rateslib.com/script/demo/"><img src="_static/rateslib_js_logo_small.png" alt="Rateslib-JS" width="170" height="37"></a>
     </div>
     <div class="flex-item-right">
       <a href="https://rateslib.com/script/demo/">rateslib-js</a> allows integration with JavaScript/TypeScript via PyOdide, which is
        a Python interpreter for web browsers. The demo page gives a browser build where the code is shown in source,
        and integrates with Vue.js. Also see <a href="https://www.linkedin.com/pulse/rateslib-javascript-via-pyodide-rateslib-qpmpf/">this LinkedIn article example</a>.
     </div>
   </div>
   <div class="flex-container-line">
     <div class="flex-item">
        <a href="https://rateslib.com/rs/latest/rateslib/"><img src="_static/rateslib_rs_logo_small.png" alt="Rateslib-Rust" width="170" height="37"></a>
     </div>
     <div class="flex-item-right">
       <a href="https://rateslib.com/rs/latest/rateslib/">rateslib-rs</a> is the lower level codebase written in Rust, with PyO3 bindings,
        providing performant solutions for some of the Python classes and methods.
     </div>
   </div>

.. raw:: html

   <div class="clear" style="text-align: center; padding: 1em 0em 1em;"></div>


*Curve construction is simple but has huge flexibility*
--------------------------------------------------------

Multiple interpolation modes are offered by default and the generalised process for curve
solving means very specific pricing artefacts can be accurately modelled with the
correct formulations. The framework is accessible and requires minimal configuration.

.. container:: twocol

   .. container:: leftside40

      .. code-block:: python

         usd_curve = Curve(
             nodes={...},
             convention="act360",
             calendar="nyc",
             interpolation="log_linear",
             id="sofr",
         )
         solver = Solver(
             curves=[usd_curve],
             instruments=[...],
             weights=[...],
             s=[...],
         )

   .. container:: rightside60

      .. image:: _static/ptirds_00_00.png
         :align: center
         :alt: See Cookbook: Single currency curve replication
         :height: 220
         :width: 380
         :target: z_ptirds_curve.html

.. raw:: html

   <div class="clear"></div>

*API is designed for users with full documentation*
-----------------------------------------------------------------

Although any fixed income library uses complex mathematical processes, the API has been
carefully designed to provide a workflow that is very intuitive. In the case of using it
for small scale learning items often few parameters and arguments are required.
For larger series of curves and more complicated object oriented
associations the API signature does not materially change. Best practice is demonstrated in
documentation examples.

.. code-block:: python

   xcs = XCS(
       effective=dt(2022, 2, 14), termination="15M",
       notional=100e6, float_spread=-10.25,
       spec="eurusd_xcs", curves=[...],
   )  # Create a EUR/USD Cross-Ccy Swap

   xcs.rate(solver=solver)
   <Dual: -14.294203, [...], [...]>

   xcs.npv(solver=solver, base="eur")
   <Dual: -50,073.295467, [...], [...]>

*Wide range of fixed income Instruments available*
----------------------------------------------------

The most recent version of *rateslib* contains the main *Instruments* that
dominate linear fixed income products. The large array of input parameters for these gives scope
to fully capture the nuances of these products across sectors and geographic regions,
capturing aspects like trading calendars, day count conventions, payment delays, etc. New
specifications and calendars are continually being added as users enquire.

A good example is a **US Treasury Bond**:

.. ipython:: python

   ust = FixedRateBond(
       effective=dt(2023, 8, 15), termination=dt(2033, 8, 15),
       fixed_rate=3.875, spec="us_gb"
   )  # Create a US-Treasury bond
   ust.price(ytm=4.0, settlement=dt(2025, 2, 14))
   ust.duration(ytm=4.0, settlement=dt(2025, 2, 14), metric="risk")


*Minimal dependencies to other Python libraries*
--------------------------------------------------

The dependencies are to **NumPy**, **Pandas**, and **Matplotlib**. *Rateslib* does
not have any dependencies to any automatic
differentiation libraries, such as PyAudi or JAX, preferring initially to use its
own forward mode module.

The test coverage is very high.


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
    i_developers.rst
    i_purchase.rst
