.. _instruments-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Instruments
**********************

The following groups outline the different *Instrument* types
offered by *rateslib*.

Common methods which all *Instruments* strive to possess are
``rate()``, ``npv()``, ``analytic_delta``, ``delta()``, ``gamma()``,
``cashflows()`` and ``cashflows_table``.

:ref:`Securities<securities-doc>`

.. inheritance-diagram:: rateslib.instruments.FixedRateBond rateslib.instruments.FloatRateNote rateslib.instruments.Bill rateslib.instruments.IndexFixedRateBond rateslib.instruments.BondFuture
   :private-bases:
   :parts: 1

.. toctree::
    :maxdepth: 0
    :titlesonly:

    e_securities.rst
    e_singlecurrency.rst
    e_multicurrency.rst
    e_combinations.rst