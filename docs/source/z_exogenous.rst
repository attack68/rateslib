.. _cook-exogenous-doc:

.. ipython:: python
   :suppress:

   from rateslib import FXRates, Curve, Solver, IRS
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

What are Exogenous Variables and Exogenous Sensitivities?
*****************************************************************

**Endogenous variables**

Being a fixed income library, there are some *variables* that are **endogenous** to *rateslib* -
meaning they are created internally and used throughout its internal calculations. These are
often easy to spot. For example when creating an *FXRates* object you will notice the user input
for FX rate information is just expressed with regular *floats*, but *rateslib* internally creates
dual number exposure to these variables.

.. ipython:: python

   fxr = FXRates({"eurusd": 1.10, "gbpusd": 1.25}, settlement=dt(2000, 1, 1))
   fxr.rate(pair="eurgbp")

Similarly, when building *Curves* and calibrating them with a *Solver*, *rateslib* structures
all its parameters internally, so that it can calculate :meth:`~rateslib.solver.Solver.delta` and
:meth:`~rateslib.solver.Solver.gamma` later without any further user input.

.. ipython:: python

   curve = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="curve")
   solver = Solver(
       curves=[curve],
       instruments=[IRS(dt(2000, 1, 1), "6m", "S", curves=curve)],
       s=[2.50]
   )
   IRS(dt(2000, 1, 1), "6m", "S", fixed_rate=3.0).npv(curves=curve)

**Exogneous variables**

**Exogenous** variables are those created dynamically by a user. The only reason one would typically
do this is to create a baseline for measuring some financial sensitivity.