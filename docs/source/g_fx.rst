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

Please review the documentation for :ref:`FXRates<fxr-doc>` first, before
proceeding to review the documentation for  :ref:`FXForwards<fxr-doc>`.

.. toctree::
   :maxdepth: 0
   :titlesonly:

   f_fxr.rst
   f_fxf.rst


When we value NPV, what is `base`?
-------------------------------------

One of the most important aspects to keep track of when valuing
:meth:`Instrument.npv()<rateslib.instruments.BaseMixin.npv>` is that
of the currency in which it is displayed. This is the ``base``
currency it is displayed in.

In order to provide a flexible, but minimal, UI *base* does not need to
be explicitly set to get the results one expects. The arguments needed for the
*npv* method are:

  ``curves``, ``solver``, ``fx``, ``base``, ``local``

All of these arguments are optional since one might typically be inferred
from another. This creates some complexity particularly when *base* is
not given and it might be inferred from others, or when *base* is given
but it conflicts with the *base* associated with other objects.

**The local argument**

``local`` can, at any time, be set to *True* and this will return a dict
containing a currency key and a value. By using this we keep track
of the currency of each *Leg* of the *Instrument*. This is important for
risk sensitivities and is used internally, especially for multi-currency instruments.

.. ipython:: python

   curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
   fxr = FXRates({"usdeur": 0.9, "gbpusd": 1.25}, base="gbp", settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={"usdusd": curve, "eureur": curve, "gbpgbp": curve, "eurusd": curve, "gbpusd": curve},
       base="eur",
   )
   solver = Solver(
       curves=[curve],
       instruments=[IRS(dt(2022, 1, 1), "1y", "a", curves=curve)],
       s=[4.109589041095898],
       fx=fxf,
   )

.. ipython:: python

   nxcs = NonMtmXCS(dt(2022, 2, 1), "6M", "A", currency="eur", leg2_currency="usd")
   nxcs.npv(curves=[curve]*4, fx=fxf, local=True)
   nxcs.npv(curves=[curve]*4, fx=fxf, base="usd")


Best Practice
***************

If you want to return an *npv* value in local currency (or in *Leg1* currency for multi-currency instruments),
then you do **not** need to supply ``base`` or ``fx`` arguments. However, to be explicit,
*base* can also be specified.

.. ipython:: python

   irs = IRS(dt(2022, 2, 1), "6M", "A", currency="usd", fixed_rate=4.0, curves=curve)
   irs.npv(solver=solver)              # USD is local currency default, solver.fx.base is EUR.
   irs.npv(solver=solver, base="usd")  # USD is explicit, solver.fx.base is EUR.

To calculate a value in another non-local currency supply an ``fx`` object and
specify the ``base``. It is **not** good practice to supply ``fx`` as numeric since this
can result in errors (if the exchange rate is given the wrong way round (human error))
and it does not preserve AD or any FX sensitivities. *base* is inferred from the
*fx* object so the following are all equivalent.

.. ipython:: python

   irs.npv(fx=fxr)                 # GBP is fx's base currency
   irs.npv(fx=fxr, base="gbp")     # GBP is explicitly specified
   irs.npv(fx=fxr, base=fxr.base)  # GBP is fx's base currency

Technical rules
****************

If ``base`` is not given it will be inferred from one of two objects;

- either it will be inferred from the provided ``fx`` object,
- or it will be inferred from the *Leg* or from *Leg1* of an *Instrument*.

``base`` will **not** be inherited from a second layer inherited object. I.e. ``base``
will not be set equal to the base currency of the ``solver.fx`` associated object.

.. image:: _static/base_inherit.png
  :alt: Inheritance map for base
  :width: 350

.. list-table:: Possible argument combinations supplied and rateslib return.
   :widths: 66 5 5 12 12
   :header-rows: 1

   * - **Case and Output**
     - ``base``
     - ``fx``
     - ``solver`` with *fx*
     - ``solver`` without *fx*
   * - ``base`` **is explicit**
     -
     -
     -
     -
   * - Returns if *currency* and ``base`` are available in ``fx`` object, otherwise
       raises.
     - X
     - X
     -
     -
   * - Returns and warns about best practice.
     - X
     - (numeric)
     -
     -
   * - Returns if *currency* and ``base`` are available in ``fx`` object, otherwise
       raises.
     - X
     -
     - X
     -
   * - Returns if *currency* and ``base`` are available in ``fx`` object, otherwise
       raises. Will warn if ``fx`` and ``solver.fx`` are not the same object.
     - X
     - X
     - X
     -
   * - Returns if ``base`` aligns with local currency, else raises.
     - X
     -
     -
     -
   * - Returns if ``base`` aligns with local currency, else raises.
     - X
     -
     -
     - X
   * - ``base`` **is inferred** and logic reverts to above cases.
     -
     -
     -
     -
   * - Returns inferring ``base`` from ``fx`` object.
     - <-
     - X
     -
     -
   * - Returns inferring ``base`` from ``fx`` object. Warns if ``fx`` and
       ``solver.fx`` are not the same object.
     - <-
     - X
     - X
     -
   * - Returns inferring ``base`` from ``fx`` object.
     - <-
     - X
     -
     - X
   * - Returns inferring ``base`` as *Leg* or *Leg1* local currency.
     - (local)
     -
     - X
     -
   * - Returns inferring ``base`` as *Leg* or *Leg1* local currency.
     - (local)
     -
     -
     - X
   * - Returns inferring ``base`` as *Leg* or *Leg1* local currency.
     - (local)
     -
     -
     -

Examples
**********

We continue the examples above using the USD IRS created and consider possible *npvs*:

.. ipython:: python

   def npv(irs, curves=None, solver=None, fx=None, base=None):
      try:
         _ = irs.npv(curves, solver, fx, base)
      except Exception as e:
         _ = str(e)
      return _

.. ipython:: python
   :okwarning:

   # The following are all explicit EUR output
   npv(irs, base="eur")          # Error since no conversion rate available.
   npv(irs, base="eur", fx=fxr)  # Takes 0.9 FX rate from object.
   npv(irs, base="eur", fx=2.0)  # UserWarning and no fx Dual sensitivities.
   npv(irs, base="eur", solver=solver)  # Takes 0.95 FX rates from solver.fx
   npv(irs, base="eur", fx=fxr, solver=solver)  # Takes 0.9 FX rate from fx

   # The following infer the base
   npv(irs)                         # Base is inferred as local currency: USD
   npv(irs, fx=fxr)                 # Base is inferred from fx: GBP
   npv(irs, fx=fxr, base=fxr.base)  # Base is explicit from fx: GBP
   npv(irs, fx=fxr, solver=solver)  # Base is inferred from fx: GBP. UserWarning for different fx objects
   npv(irs, solver=solver)          # Base is inferred as local currency: USD
   npv(irs, solver=solver, fx=solver.fx)  # Base is inferred from solver.fx: EUR
