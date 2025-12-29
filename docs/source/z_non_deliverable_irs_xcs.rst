.. _cook-ndirs-doc:

.. ipython:: python
   :suppress:

   from rateslib import dt, Solver, Curve, FXRates, FXForwards, IRS, NDXCS, defaults
   import matplotlib.pyplot as plt


Non-Deliverable IRS and XCS: EM Markets
*****************************************

*Rateslib* v2.5 introduced non-deliverable *IRS* and *XCS*. This page exemplifies how to use
these objects to calibrate *Curves* in those markets. Specifically here we will use ND-IRS and
NDXCS to calibrate Indian Rupee *Curves*.

  **Key Points**

  - A USD `Curve` is established in the normal way using US instruments.
  - An `FXForwards` market is proposed (uncalibrated) between USD and INR.
  - The suite of non-deliverable *Instruments* are constructed and used for calibration.

The Deliverable US Market
-------------------------

First, we need to establish the baseline US market and the SOFR curve. This is no different to
any other tutorial on the matter, so we quickly do this with some of the following SOFR swap data.
Just for some variety, this *Curve* will be interpolated with a log-cubic DF spline.

.. ipython:: python

   usd = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act360",
       calendar="nyc",
       interpolation="spline",
       id="sofr"
   )
   us_solver = Solver(
       curves=[usd],
       instruments=[
           IRS(dt(2025, 12, 31), "1y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "2y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "3y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "4y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "5y", spec="usd_irs", curves="sofr"),
       ],
       s=[3.434, 3.302, 3.314, 3.359, 3.416],
   )

The Components for the ``FXForwards``
----------------------------------------

We are going to use just the 1Y through 5Y instruments, as demonstration, to calibrate the
INR market. So our local FBIL Overnight Mumbai Interbank Outright Rate (FBIL-O/N MIBOR) is the
following:

.. ipython:: python

   inr = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act365F",
       calendar="mum",
       id="mibor-ois",
   )

In order to introduce the necessary degrees of freedom to satisfy the cross-currency market and
supply and demand we establish the basis curve:

.. ipython:: python

   inrusd = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act365F",
       calendar="all",  # <- no holiday calendar necessary for a cross-currency discount curve.
       id="inrusd",
   )

Finally we put all of the elements together to create the USDINR FXForwards market, note that
we have also input the spot USDINR FX rate here as well:

.. ipython:: python

   fxf = FXForwards(
       fx_rates=FXRates({"usdinr": 89.9812}, settlement=dt(2025, 12, 31)),
       fx_curves={"usdusd": usd, "inrinr": inr, "inrusd": inrusd},
   )

This object can now be used to forecast any USDINR rate, **but it won't be
accurate** becuase we haven't calibrated anything yet! The INR rates are currently all zero on the
2 INR *Curves*.

.. ipython:: python

   fxf.rate("usdinr", settlement=dt(2026, 12, 31))
   fxf.swap("usdinr", [dt(2025, 12, 31), dt(2026, 12, 31)])

Calibrating the Curves
-------------------------

So, we have now reached the point where we can calibrate the INR curves. We have 10 parameters /
degrees of freedom and will therefore require 10 *Instruments*. We will use 5 *NDIRS*, which will
calibrate local currency interest rates (the ``inr`` *Curve*) [Bloomberg Moniker IRSWNI1 Curncy],
and 5 *NDXCS* [Bloomberg Moniker IRUSON1 Curncy] which will effectively calibrate the
cross-currency basis.

*Rateslib* has added some ``spec`` defaults for the purpose of this article, but the keyword
arguments used can be directly observed below:

.. ipython:: python

   defaults.spec["inr_ndirs"]
   defaults.spec["inrusd_ndxcs"]

To calibrate we must include the previous US :class:`~rateslib.solver.Solver`, which contains
the mapping to the constructed US SOFR *Curve*, and we specify the *Instruments* and the live
market data rates.

.. ipython


.. ipython:: python

   inr_solver = Solver(
       pre_solvers=[us_solver],
       curves=[inr, inrusd],
       instruments=[
           IRS(dt(2025, 12, 30), "1Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "2Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "3Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "4Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "5Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           NDXCS(dt(2025, 12, 31), "1Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "2Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "3Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "4Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "5Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
       ],
       s=[
           5.47, 5.5525, 5.715, 5.835, 5.925, #  <-  IRS rates
           6.375, 6.335, 6.415, 6.535, 6.595  #  <-  XCS rates
       ],
       fx=fxf,
   )

What is interesting to note about this particular *Solver* configuration is that nowhere does the
*'inrusd'* discount *Curve* enter any *Instrument* specification. Since these *Instruments* have
non-deliverable cashflows every discount *Curve* is the USD SOFR *Curve*. The key pricing component
here is the ``fx=fxf`` object, which is a **pricing** parameter that *is* needed and is passed to
all *Instruments*, and of course it derives forward FX rates using the *'inrusd'* *Curve* so
everything is calibrated accurately.

The datasource (**DS**) for these prices also gives (wide) financial bid/ask for FX swaps and FX forwards.
We can compare these with the :class:`~rateslib.fx.FXForwards` we have constructed through *rateslib* (**RL**)
calibration.

.. ipython:: python
   :suppress:

   from pandas import DataFrame
   from rateslib.dual.utils import _dual_float
   df = DataFrame({
       "tenor": ["1y", "2y", "3y", "4y", "5y"],
       "DS forward": [92.5112, 95.4512, 98.3212, 101.3112, 104.7912],
       "DS swap": [25300, 54700, 83400, 113300, 148100],
       "RL forward": [
            _dual_float(fxf.rate("usdinr", dt(2026, 12, 31))),
            _dual_float(fxf.rate("usdinr", dt(2027, 12, 31))),
            _dual_float(fxf.rate("usdinr", dt(2028, 12, 29))),
            _dual_float(fxf.rate("usdinr", dt(2029, 12, 31))),
            _dual_float(fxf.rate("usdinr", dt(2030, 12, 31))),
       ],
       "RL swap": [
            _dual_float(fxf.swap("usdinr", [dt(2025, 12, 31), dt(2026, 12, 31)])),
            _dual_float(fxf.swap("usdinr", [dt(2025, 12, 31), dt(2027, 12, 31)])),
            _dual_float(fxf.swap("usdinr", [dt(2025, 12, 31), dt(2028, 12, 29)])),
            _dual_float(fxf.swap("usdinr", [dt(2025, 12, 31), dt(2029, 12, 31)])),
            _dual_float(fxf.swap("usdinr", [dt(2025, 12, 31), dt(2030, 12, 31)])),
       ]
   })

.. ipython:: python

   df

Lets have a look at the calibrate *Curves* thus far:

.. ipython:: python

   usd.plot("1b", comparators=[inr, inrusd], labels=["SOFR", "ON/MIBOR", "ON/MIBOR+Basis"])

.. plot::

   from rateslib import dt, Solver, Curve, FXRates, FXForwards, IRS, NDXCS
   import matplotlib.pyplot as plt

   usd = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act360",
       calendar="nyc",
       interpolation="spline",
       id="sofr"
   )
   us_solver = Solver(
       curves=[usd],
       instruments=[
           IRS(dt(2025, 12, 31), "1y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "2y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "3y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "4y", spec="usd_irs", curves="sofr"),
           IRS(dt(2025, 12, 31), "5y", spec="usd_irs", curves="sofr"),
       ],
       s=[3.434, 3.302, 3.314, 3.359, 3.416],
   )

   inr = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act365F",
       calendar="mum",
       id="mibor-ois"
   )

   inrusd = Curve(
       nodes={
           dt(2025, 12, 29): 1.0,
           dt(2026, 12, 29): 1.0,
           dt(2027, 12, 29): 1.0,
           dt(2028, 12, 29): 1.0,
           dt(2029, 12, 29): 1.0,
           dt(2031, 1, 7): 1.0,
       },
       convention="Act365F",
       calendar="all",  # <- no holiday calendar necessary for a cross-currency discount curve.
       id="inrusd"
   )

   fxf = FXForwards(
       fx_rates=FXRates({"usdinr": 89.9812}, settlement=dt(2025, 12, 31)),
       fx_curves={"usdusd": usd, "inrinr": inr, "inrusd": inrusd},
   )

   inr_solver = Solver(
       pre_solvers=[us_solver],
       curves=[inr, inrusd],
       instruments=[
           IRS(dt(2025, 12, 30), "1Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "2Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "3Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "4Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           IRS(dt(2025, 12, 30), "5Y", spec="inr_ndirs", curves=["mibor-ois", "sofr"]),
           NDXCS(dt(2025, 12, 31), "1Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "2Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "3Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "4Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
           NDXCS(dt(2025, 12, 31), "5Y", spec="inrusd_ndxcs", curves=[None, "sofr", "sofr", "sofr"]),
       ],
       s=[5.47, 5.5525, 5.715, 5.835, 5.925, 6.375, 6.335, 6.415, 6.535, 6.595],
       fx=fxf,
   )

   fig, ax, line = usd.plot("1b", comparators=[inr, inrusd], labels=["SOFR", "ON/MIBOR", "ON/MIBOR+Basis"])
   plt.show()
   plt.close()
