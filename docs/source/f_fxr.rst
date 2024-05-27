.. _fxr-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import *
   from datetime import datetime as dt

***********************
FX Spot Rates
***********************

This documentation page discusses the methods of the
:class:`~rateslib.fx.FXRates` class which are summarised below:

.. autosummary::
   rateslib.fx.FXRates
   rateslib.fx.FXRates.rate
   rateslib.fx.FXRates.rates_table
   rateslib.fx.FXRates.convert
   rateslib.fx.FXRates.convert_positions
   rateslib.fx.FXRates.positions
   rateslib.fx.FXRates.update
   rateslib.fx.FXRates.restate
   rateslib.fx.FXRates.to_json
   rateslib.fx.FXRates.from_json

Introduction
------------

:class:`~rateslib.fx.FXRates` classes are initialised straightforwardly with a
given set of FX rates.
The optional ``settlement`` argument is only used in conjunction
with :class:`~rateslib.fx.FXForwards`
specification. The cross multiplications
to derive all FX rates from the stated FX market are performed internally.

.. ipython:: python

   fxr = FXRates(
       fx_rates={"eurusd": 1.1, "eursek": 10.85, "noksek": 1.05, "gbpusd":1.25},
       base="usd",
   )
   fxr.rates_table()

The :class:`~rateslib.fx.FXRates` class is also used when one has an ``Instrument``
in one currency but the results of calculations are preferred in another, say, when
accounting currency is something other than the currency of underlying ``Instrument``.
For example, below we create a :class:`~rateslib.curves.Curve` and a EUR
:class:`~rateslib.instruments.IRS`, and calculate some metrics expressed in that
currency.

.. ipython:: python

   curve = Curve(
       nodes={dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 0.99, dt(2023, 1, 1): 0.97},
       id="estr",
   )
   swap = IRS(dt(2022, 1, 1), "1Y", "A", fixed_rate=2.00, currency="eur")
   swap.npv(curve)
   swap.analytic_delta(curve)

The :class:`~rateslib.fx.FXRates` class defined above can be used to directly return
the above metrics in the specified base currency (USD).

.. ipython:: python

   swap.npv(curve, fx=fxr)
   swap.analytic_delta(curve, fx=fxr)
   swap.cashflows(curve, fx=fxr).transpose()

Or, other currencies too, that are non-base, can also be displayed upon request.

.. ipython:: python

   swap.npv(curve, fx=fxr, base="nok")

.. _fx-dual-doc:

Sensitivity Management
----------------------

This object does not only create an FX :meth:`~rateslib.fx.FXRates.rates_table`,
it also performs calculations
and determines sensitivities, using automatic differentiation, to the FX rates that
are given as the parameters in the construction. For example, in the above
construction the EURSEK and NOKSEK rates are given, as *majors*.
The EURNOK exchange rate, is a *cross*, and being derived from those means it
will demonstrate that dependency to those two, whilst the EURSEK rate
will demonstrate only direct one-to-one dependency with the quoted EURSEK rate.

.. ipython:: python

   fxr.rate("eursek")
   fxr.rate("eurnok")

In a similar manner cashflows, that are converted from one currency to another also
maintain sensitivity calculations stored within their :class:`~rateslib.dual.Dual`
number specification.

.. ipython:: python

   sek_value = fxr.convert(100, "eur", "sek")
   sek_value

Interpreting Dual Values
************************

The above value has an *"fx_eursek"* dual value of 100 (SEK). This means that for the
EURSEK rate to increase by 1.0 from 10.85 to 11.85 the base (SEK) value would
increase by 100, from 1,085 SEK to 1,185 SEK. In this case this is exact, but the
figure of *"100"* represents an instantaneous derivative. When dealing with reverse
exposures (i.e SEKEUR) this becomes apparent.

.. ipython:: python

   eur_value = fxr.convert(1085, "sek", "eur")
   eur_value

Now when EURSEK increases to 11.85 the new *"eur_value"* would actually be 91.56 EUR.
This is **not** (100-9.2166=) 90.78. But this sensitivity is applicable on an
infinitesimal basis.

Conversion Methods
******************

By interpreting and storing values with FX sensitivities the underlying true positions
are maintained.
A 100 EUR cash position *valued* as 1,085 SEK, is not the same as a 1,085 SEK
cash position when considering financial risk exposures. Therefore the methods
:meth:`~rateslib.fx.FXRates.convert`, :meth:`~rateslib.fx.FXRates.convert_positions`
and :meth:`~rateslib.fx.FXRates.positions` exist to
seamlessly transition between the different representations.

.. ipython:: python

   cash_positions = fxr.positions(sek_value, base="sek")
   cash_positions

And the cash positions can be converted into any base representation currency.

.. ipython:: python

   eur_value = fxr.convert_positions(cash_positions, base="eur")
   eur_value

Updating
--------

Once an :class:`~rateslib.fx.FXRates` class has been instantiated it may then be
associated with
other objects, such as an :class:`~rateslib.fx.FXForwards` class.

.. note::

   It is **best practice**
   not to create further :class:`~rateslib.fx.FXRates` instances but
   to **update** the existing
   ones instead.
   Please review the documentation for :meth:`~rateslib.fx.FXRates.update` for
   further details.
