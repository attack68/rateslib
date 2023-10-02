.. _instruments-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Instruments
**********************

The following groups outline the different *Instrument* types
offered by *rateslib*. The common methods which all *Instruments* strive to possess are
``rate()``, ``npv()``, ``analytic_delta()``, ``delta()``, ``gamma()``,
``cashflows()`` and ``cashflows_table()``.

Securities
----------

Link to the section on :ref:`securities<securities-doc>`, which comprise *Bonds, Bills* and
*Bond Futures*.

Single Currency Derivatives
---------------------------

Link to the section on :ref:`single currency derivatives<singlecurrency-doc>`, giving objects
like *IRSs*, *SBSs*, *ZCISs* and *ZCSs* etc.


Multi-Currency Derivatives
--------------------------

Link to the section on :ref:`multi-currency derivatives<multicurrency-doc>`. This allows
*FXSwaps*, *Cross-Currency Swaps* and *FX Exchanges*.

Utilities and Instrument Combinations
-------------------------------------

Link to the section on :ref:`utilities and instrument combinations<combinations-doc>`. This
allows things like *Spread trades*, *Butterflies*, *Portfolios* and a *Value* for a *Curve*.


.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    e_securities.rst
    e_singlecurrency.rst
    e_multicurrency.rst
    e_combinations.rst
