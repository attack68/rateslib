.. _fx-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import FXRates, FXForwards
   from rateslib.curves import Curve
   from rateslib.instruments import Value
   from rateslib.solver import Solver
   from datetime import datetime as dt
   import math

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

FX Rates
**********

The :class:`~rateslib.fx.FXRates` class is a **simple object** designed to record
and store FX exchange rate values for a particular settlement date, i.e. it is
expected to be used for spot FX rates, and for value conversion from one currency to another.

.. ipython:: python

   fxr = FXRates(fx_rates={"audusd": 0.62}, settlement=dt(2003, 4, 7))
   fxr.rates_table()

:ref:`Read more here<fxr-doc>` about these and other methods of the *FXRates* class .

FX Forwards
*************

The :class:`~rateslib.fx.FXForwards` class is a more **comprehensive** object
that uses a mapping of interest rate curves to derive forward FX rates for any settlement date.

As a brief, blast from the past example, we replicate John Hull's *Options, Futures and
Other Derivatives (fifth edition: equation 3.13)*. He derives the 2y forward rate for AUDUSD FX,
via interest rate parity,
with known continuously compounded domestic and foreign interest rates, and the spot (although
technically immediate) FX rate as above.

.. math::

   F_0 &= S_0 e^{(r_{rhs}-r_{lhs}) T} \\
   0.645303 &= 0.6200 e^{(0.07 - 0.05) \times 2}

.. ipython:: python

   aud = Curve({dt(2003, 4, 7): 1.0, dt(2005, 4, 7): math.exp(-0.05*2)}, id="aud")
   usd = Curve({dt(2003, 4, 7): 1.0, dt(2005, 4, 7): math.exp(-0.07*2)}, id="usd")
   fxf = FXForwards(
      fx_rates=fxr,
      fx_curves={"audaud": aud, "usdusd": usd, "audusd": aud},
   )

The discount factors (DFs) on the currency *Curves* have been specified directly
to match the continuously compounded zero rates in accordance with the specification
in Hull's book. We could also have calibrated these with a *Solver* using the
:class:`~rateslib.instruments.Value` *Instrument* with a ``metric`` of *'cc_zero_rate'*.

The value :math:`F_0` is then directly available, along with forward rates on
any chosen ``settlement``.

.. ipython:: python

   fxf.rate("audusd", dt(2005, 4, 7))

:ref:`Read more here<fxf-doc>` about the *FXForwards* class.

FX Volatility
***************

*Rateslib* also has some support for *FX volatility*. :ref:`Read more here<fx-volatility-doc>`.

.. toctree::
   :hidden:
   :maxdepth: 0
   :titlesonly:

   f_fxr.rst
   f_fxf.rst
