.. _securities-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Securities
**********************

``Securities`` are generally one-leg instruments which have
been packaged to provide specific methods relevant to their
nature. For example bonds have yield-to-maturity and accrued interest
for example.

.. inheritance-diagram:: rateslib.instruments.FixedRateBond rateslib.instruments.FloatRateNote rateslib.instruments.Bill rateslib.instruments.IndexFixedRateBond rateslib.instruments.BondFuture
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.FixedRateBond
   rateslib.instruments.FloatRateNote
   rateslib.instruments.Bill
   rateslib.instruments.IndexFixedRateBond
   rateslib.instruments.BondFuture


Fixed Rate Bond
****************

Fixed rate bonds can be constructed and priced with traditional metrics.
The following example is taken from the UK DMO's documentation.

.. ipython:: python

   bond = FixedRateBond(
       effective=dt(1995, 1, 1),
       termination=dt(2015, 12, 7),
       frequency="S",
       convention="ActActICMA",
       fixed_rate=8.0,
       ex_div=7,
       settle=1,
       calendar="ldn",
   )

The ``price`` in a *dirty* and *clean* sense related by the ``accrued`` is visible
below for a ``ytm`` (yield-to-maturity) of 4.445%.

.. ipython:: python

   bond.price(
      ytm=4.445,
      settlement=dt(1999, 5, 27),
      dirty=True
   )
   bond.ex_div(dt(1999, 5, 27))
   bond.accrued(dt(1999, 5, 27))
   bond.price(
      ytm=4.445,
      settlement=dt(1999, 5, 27),
      dirty=False
   )

Bonds can also be priced by a discount :class:`~rateslib.curves.Curve`. Since the
bond has settlement timeframe of 1 business day this will be one business day
after the initial node date of the curve.

.. ipython:: python

   bond_curve = Curve({dt(1999, 5, 26): 1.0, dt(2015, 12, 7): 0.483481})
   bond.rate(bond_curve, metric="dirty_price")
   bond.rate(bond_curve, metric="ytm")
