.. _cook-stubs-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Pricing IBOR Interpolated Stub Periods
******************************************************

**Stubs** on derivatives can often cause problems in derivatives pricing. The *rateslib* framework
developed is designed to make the UI and pricing process simple. That is, an IRS, for example,
has **two** *Legs* and each *Leg* might have two *Curves* for pricing; a forecasting *Curve* to
project fixings (which is not necessary for the fixed leg), and a discounting *Curve* to discount
the projected cashflows.

This all works well except for IBOR stub periods, which might need two forecasting curves on a *Leg*
to project two different IBOR rates and interpolate them to derive the resultant fixing for the
:class:`~rateslib.periods.FloatPeriod`.

If we setup a pricing model we can explore the possibilities. This model creates Euribor 1M and
3M *Curves* and a separate ESTR discounting *Curve*.

.. ipython:: python

   solver = Solver(
       curves=[
          Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="estr"),
          Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eur1m"),
          Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eur3m"),
       ],
       instruments=[
           IRS(dt(2022, 1, 1), "1Y", spec="eur_irs", curves="estr"),
           IRS(dt(2022, 1, 1), "1Y", spec="eur_irs1", curves=["eur1m", "estr"]),
           IRS(dt(2022, 1, 1), "1Y", spec="eur_irs3", curves=["eur3m", "estr"]),
       ],
       s=[1.5, 1.6, 1.7],
       instrument_labels=["1Ye", "1Y1s", "1Y3s"],
       id="eur rates"
   )

We now create an :class:`~rateslib.instruments.IRS` which has 2 month **stub periods** at the
front and back of the *Instrument*.

.. ipython:: python

   irs = IRS(
       effective=dt(2022, 1, 15),
       termination=dt(2022, 11, 15),
       front_stub=dt(2022, 3, 15),
       back_stub=dt(2022, 9, 15),
       frequency="Q",
       fixed_rate=1.5,
       currency="eur",
       leg2_fixing_method="ibor"
   )

If we price and risk this swap naively using the 3M IBOR curve the stub periods will be
calculated from that 3M curve directly, and there is no dependence at all to the 1M curve.

.. ipython:: python

    irs.rate(curves=["eur3m", "estr"], solver=solver)
    irs.delta(curves=["eur3m", "estr"], solver=solver)
    irs.cashflows(curves=["eur3m", "estr"], solver=solver)

However, it is also possible, only in the case of an *"ibor"* ``fixing_method``, to supply a *dict*
of forecasting curves, from which it will interpolate the fixing using the maturity date of the
tenor fixings and the end date of the period. In this format the *keys* of the dict are the
IBOR tenors available, e.g. *"3m"* and the *values* of the dict represent the *Curve* objects or
the *Curve* str ids from which will identify the *Curves* to be extracted from the *Solver*.

.. ipython:: python

    irs.rate(curves=[{"3m": "eur3m", "1m": "eur1m"}, "estr"], solver=solver)
    irs.delta(curves=[{"3m": "eur3m", "1m": "eur1m"}, "estr"], solver=solver)
    irs.cashflows(curves=[{"3m": "eur3m", "1m": "eur1m"}, "estr"], solver=solver)

Notice that in this case the relevant risk sensitivity exposure has been measured against the
1M curve to which the *IRS* has some direct dependence.
