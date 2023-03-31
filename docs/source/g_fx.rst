.. _fx-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

************
FX
************

The ``rateslib.fx`` module is also integral to a fixed income library.
It allows PVs and IR
risk sensitivities calculations to be expressed in chosen currencies - to provide
consistency across multi-currency portfolios. The ability to measure FX rate risk
sensitivities is also provided. Additionally the construction of multi-currency
instruments and multi-currency curves requires FX forward rate calculation.

The two main classes provided are distinct.

- :class:`~rateslib.fx.FXRates` is a **much simpler object** designed to record
  and store FX exchange rate values for a particular settlement date, i.e. it is
  expected to be used for spot FX rates.
- :class:`~rateslib.fx.FXForwards` is a more **complicated** class that requires a
  mapping of interest rate curves to derive forward FX rates for any settlement date.

Please review the documentation for :ref:`FXRates<fxr-doc>` first, before
proceeding to review the documentation for  :ref:`FXForwards<fxr-doc>`.

.. toctree::
   :maxdepth: 0
   :titlesonly:

   f_fxr.rst
   f_fxf.rst
