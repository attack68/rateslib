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


What is `base`?
----------------

The common  arguments needed for the
:meth:`Instrument.npv()<rateslib.instruments.BaseMixin.npv>` and similarly
derived methods are:

  [``curves``, ``solver``, ``fx``, ``base``]

All of these arguments are optional since one might typically be inferred
from another. This creates some complexity particularly when *base* is
not given and it might be inferred from others, or when *base* is given
but it conflicts with the *base* associated with other objects.

If ``base`` is not given it will be inferred from one of two objects;

- either it will be inferred from the provided ``fx`` object,
- or it will be inferred from the *Leg* or from *Leg1* of an *Instrument*.

``base`` will not be inherited from a second layer inherited object. I.e. ``base`` will
not be set equal to the base currency of the ``solver.fx`` associated object.

.. image:: _static/base_inherit.png
  :alt: Inheritance map for base
  :width: 400

.. list-table:: Arguments given and inferred result
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



1) Is ``base`` given explicitly?
*********************************

**YES**: Use this as the input directly.

This will raise errors if an FX conversion cannot be explicitly calculated.

.. ipython:: python

   curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.96}, id="curve")
   irs = IRS(dt(2022, 2, 1), "6M", "A", currency="usd", fixed_rate=4.0)
   try:
       irs.npv(curves=curve, base="eur")
   except ValueError as e:
       print(e)

This will overwrite the base currency on a *FXRates* or *FXForwards* object.

.. ipython:: python

   fxr = FXRates({"eurusd": 1.1, "gbpusd": 1.25}, base="gbp")
   irs.npv(curves=curve)
   irs.npv(curves=curve, fx=fxr, base="eur")

**NO**: Goto 2)

2) Is an ``fx`` object (*FXRates* or *FXForwards*) given explicitly?
*********************************************************************

**YES**: then ``base`` is inferred from this object.

.. ipython:: python

   irs.npv(curves=curve, fx=fxr)  # fxr has base 'gbp' and the output is in GBP.

If a ``solver`` is given which also contains an ``fx`` attribute then the *Solver's*
attribute is ignored in favour of the explicit ``fx`` object provided. This will
raise a *UserWarning*, however.

.. ipython:: python
   :okwarning:

   solver = Solver(
       curves=[curve],
       instruments=[IRS(dt(2022, 1, 1), "1y", "a", curves=curve)],
       s=[4.109589041095898],
       fx=FXRates({"usdeur": 1.5}, base="eur"),
   )
   irs.npv(curves=curve, solver=solver, fx=fxr)

