.. _cook-multicsadisc-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from rateslib.dual import Dual, gradient
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context
   defaults.convention = "Act360"
   assert defaults.convention.lower() == "act360"

MultiCsaCurves have discontinuous derivatives
******************************************************

This documentation page is written to exemplify the discontinuous nature of the
:class:`~rateslib.curves.MultiCsaCurve`. Even though the AD of *rateslib* still functions, the
definition of an *intrinsic* :class:`~rateslib.curves.MultiCsaCurve` forces discontinuity.

To set the stage, consider the **absolute value** function, :math:`abs(x)`. This function has a
continuous and calculable derivative at every point except zero.

.. ipython:: python

   x_plus_1 = Dual(1.0, ["x"], [])
   abs(x_plus_1)
   gradient(abs(x_plus_1))

   x_minus_1 = Dual(-1.0, ["x"], [])
   abs(x_minus_1)
   gradient(abs(x_minus_1))

At the point zero *rateslib* still returns a result (which contains a true derivative from one side,
but a false derivative from the other side). Users are expected to know that
automatic differentiation, calculated in this way, does not work. The **abs** function is
not AD safe over its complete domain.

.. ipython:: python

   x_zero = Dual(0.0, ["x"], [])
   abs(x_zero)
   gradient(abs(x_zero))

An intrinsic :class:`~rateslib.curves.MultiCsaCurve` has the same properties. It has real,
calculable derivatives everywhere, except when there is a crossover point from one CTD currency
to another, this is a point without a valid derivative.

Build a *MultiCsaCurve* with equal collateral currencies
----------------------------------------------------------

This section will build a :class:`~rateslib.curves.MultiCsaCurve`, for which the CTD currency
is not distinct. Either of the two collateral currencies have exactly the same 'cheapness'.
Either is valid as the CTD.

.. ipython:: python

   eur = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 1.0, dt(2010, 1, 1): 1.0})
   eurusd = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 1.0, dt(2010, 1, 1): 1.0})
   usd = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 1.0, dt(2010, 1, 1): 1.0})
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.1}, settlement=dt(2000, 1, 1)),
       fx_curves={"eureur": eur, "eurusd": eurusd, "usdusd": usd},
   )
   solver = Solver(
       curves=[eur, eurusd, usd],
       instruments=[
           IRS(dt(2000, 1, 1), "5y", spec="eur_irs", curves=eur),
           IRS(dt(2000, 1, 1), "10y", spec="eur_irs", curves=eur),
           IRS(dt(2000, 1, 1), "5y", spec="usd_irs", curves=usd),
           IRS(dt(2000, 1, 1), "10y", spec="usd_irs", curves=usd),
           XCS(dt(2000, 1, 1), "5y", spec="eurusd_xcs", curves=[eur, eurusd, usd, usd]),
           XCS(dt(2000, 1, 1), "10y", spec="eurusd_xcs", curves=[eur, eurusd, usd, usd]),
       ],
       s=[1.0, 1.5, 1.0, 1.5, 0.0, 0.0],  # <-- local ccy rates are same and no xccy basis
       instrument_labels=["5yEur", "10yEur", "5yUsd", "10yUsd", "5yXcy", "10yXcy"],
       fx=fxf,
   )

With the market setup, create the *intrinsic* :class:`~rateslib.curves.MultiCsaCurve`. This
curve discounts EUR cashflows with the cheapest to deliver of EUR and USD collateral.

.. ipython:: python

   multi_csa = fxf.curve(cashflow="eur", collateral=("eur", "usd"))
   type(multi_csa)

What happens to risk and NPV when the market moves?
------------------------------------------------------

Setup the base case for comparison. Below, an IRS is created and its NPV and risk sensitivities to
the above calibrating instruments are stated.

.. ipython:: python

   irs =  IRS(dt(2000, 1, 1), "10y", spec="eur_irs", curves=[eur, multi_csa], fixed_rate=2.0)
   irs.npv(solver=solver)
   irs.delta(solver=solver)

Now we will make USD collateral more expensive to deliver by 20bps. EUR deliverance is unchanged.

.. ipython:: python

   solver.s = [1.0, 1.5, 1.0, 1.5, -20.0, -20.0]
   solver.iterate()

And re-evaluate the risk metrics and NPV. The NPV is broadly unchanged.

.. ipython:: python

   irs.npv(solver=solver)
   irs.delta(solver=solver)

Instead of making the USD collateral more expensive relative to EUR it could be made 20bps
cheaper. The impacts for this are also shown.

.. ipython:: python

   solver.s = [1.0, 1.5, 1.0, 1.5, 20.0, 20.0]
   solver.iterate()

.. ipython:: python

   irs.npv(solver=solver)
   irs.delta(solver=solver)

By analysing these results it is clear that risk sensitivities do not always explain the NPV
changes given market movements of these instruments.
