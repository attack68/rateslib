.. _fx-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

************
FX
************

The ``rateslib.fx`` module provides ``FX`` functionality. This is a necessary
part of a fixed income library because it allows:

- consistent treatment of cashflows and values expressed in one currency relative
  to another,
- the construction of multi-currency derivatives, and of FX forward rates,
- valuation of derivatives with CSAs priced in non-local currencies,
- stored FX rate sensitivity calculations via automatic differentiation.

The two main classes provided are distinct.

- :class:`~rateslib.fx.FXRates` is a **much simpler object** designed to record
  and store FX exchange rate values for a particular settlement date, i.e. it is
  expected to be used for spot FX rates.
- :class:`~rateslib.fx.FXForwards` is a more **complicated** class that requires a
  mapping of interest rate curves to derive forward FX rates for any settlement date.

It is suggested to proceed by reading the documentation for :ref:`FXRates<fxr-doc>` first, before
then reviewing the documentation for  :ref:`FXForwards<fxr-doc>`.

*Rateslib* also supports some :ref:`FX volatility products <fx-volatility-doc>`.

.. toctree::
   :hidden:
   :maxdepth: 0
   :titlesonly:

   f_fxr.rst
   f_fxf.rst
