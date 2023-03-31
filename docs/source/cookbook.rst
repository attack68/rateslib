.. _cookbook-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

***********
Cookbook
***********

This module constructs and iteratively solves :class:`Curve` objects which are
interest rate curves defined by discount factors (DFs). They maintain
an inherent DF interpolation technique.

This module relies on the ultility modules :ref:`splines<splines-doc>`
and :ref:`dual<dual-doc>`.

.. autosummary::
   rateslib.curves.Curve
   rateslib.solver.Solver

Underspecified Curves
*********************

If the number of variables to solve for is greater than the number of calibrating
instruments to the curve (or if the choice of instruments are not well aligned with the
variables) then the :class:`Curve` is **underspecified**.

Suppose we propose a curve calibrated only by two IRS, but having node dates on central
bank policy dates throughout the year then the curve is particularly underspecified
and the numerial optimiser may yield unconstrained solutions. This is also an example
where Guass Newton will not converge so other algorithms are a better choice.


.. ipython:: python
   :okwarning:

   curve = Curve(
       nodes={
           dt(2022,1,1): 1.000,
           dt(2022,1,27): 0.999,
           dt(2022,3,17): 0.998,
           dt(2022,4,28): 0.997,
           dt(2022,6,16): 0.996,
           dt(2022,7,28): 0.995,
           dt(2022,9,22): 0.994,
           dt(2022,11,3): 0.993,
           dt(2022,12,15): 0.992,
           dt(2023,1,1): 0.991,
       },
       interpolation="log_linear",
   )
   instruments = [
       (IRS(dt(2022, 1, 1), "1D", "Q"), (curve,), {}),
       (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
   ]
   solver = Solver(
       curves=[curve],
       instruments=instruments,
       s=np.array([0.25, 2.00]),
       algorithm="levenberg_marquardt",
   )
   curve.plot("1D")

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.000,
           dt(2022,1,27): 0.999,
           dt(2022,3,17): 0.998,
           dt(2022,4,28): 0.997,
           dt(2022,6,16): 0.996,
           dt(2022,7,28): 0.995,
           dt(2022,9,22): 0.994,
           dt(2022,11,3): 0.993,
           dt(2022,12,15): 0.992,
           dt(2023,1,1): 0.991,
       },
       interpolation="log_linear",
   )
   instruments = [
       (IRS(dt(2022, 1, 1), "1D", "Q"), (curve,), {}),
       (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
   ]
   s = np.array([0.25, 2.00])
   solver = Solver(
       curves = [curve],
       instruments = instruments,
       s = s,
       algorithm = "levenberg_marquardt",
   )
   fig, ax, line = curve.plot("1D")
   plt.show()

It is advisable to create curvature constraints which serve as
regularisation.

.. ipython:: python
   :okwarning:

   curve = Curve(
       nodes={
           dt(2022,1,1): 1.000,
           dt(2022,1,27): 0.999,
           dt(2022,3,17): 0.998,
           dt(2022,4,28): 0.997,
           dt(2022,6,16): 0.996,
           dt(2022,7,28): 0.995,
           dt(2022,9,22): 0.994,
           dt(2022,11,3): 0.993,
           dt(2022,12,15): 0.992,
           dt(2023,1,1): 0.991,
       },
       interpolation="log_linear",
   )
   (m1, m2, m3, m4, m5, m6, m7, m8, m9) = (
       IRS(dt(2022,1,1),"1D", "Q"),
       IRS(dt(2022,1,27),"1D", "Q"),
       IRS(dt(2022,3,17),"1D", "Q"),
       IRS(dt(2022,4,28),"1D", "Q"),
       IRS(dt(2022,6,16),"1D", "Q"),
       IRS(dt(2022,7,28),"1D", "Q"),
       IRS(dt(2022,9,22),"1D", "Q"),
       IRS(dt(2022,11,3),"1D", "Q"),
       IRS(dt(2022,12,15),"1D", "Q"),
   )
   curvature_constraints = [
       (Fly(m1, m2, m3), (curve,), {}),
       (Fly(m2, m3, m4), (curve,), {}),
       (Fly(m3, m4, m5), (curve,), {}),
       (Fly(m4, m5, m6), (curve,), {}),
       (Fly(m5, m6, m7), (curve,), {}),
       (Fly(m6, m7, m8), (curve,), {}),
       (Fly(m7, m8, m9), (curve,), {}),
   ]
   instruments = [
       (IRS(dt(2022, 1, 1), "1D", "Q"), (curve,), {}),
       (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
   ]
   solver = Solver(
       curves = [curve],
       instruments = instruments + curvature_constraints,
       s = np.array([0.25, 2.00, 0, 0, 0, 0, 0, 0, 0]),
       weights = [1, 1, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
       algorithm = "levenberg_marquardt",
   )
   curve.plot("1D")

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.000,
           dt(2022,1,27): 0.999,
           dt(2022,3,17): 0.998,
           dt(2022,4,28): 0.997,
           dt(2022,6,16): 0.996,
           dt(2022,7,28): 0.995,
           dt(2022,9,22): 0.994,
           dt(2022,11,3): 0.993,
           dt(2022,12,15): 0.992,
           dt(2023,1,1): 0.991,
       },
       interpolation="log_linear",
   )
   (m1, m2, m3, m4, m5, m6, m7, m8, m9) = (
       IRS(dt(2022,1,1),"1D", "Q"),
       IRS(dt(2022,1,27),"1D", "Q"),
       IRS(dt(2022,3,17),"1D", "Q"),
       IRS(dt(2022,4,28),"1D", "Q"),
       IRS(dt(2022,6,16),"1D", "Q"),
       IRS(dt(2022,7,28),"1D", "Q"),
       IRS(dt(2022,9,22),"1D", "Q"),
       IRS(dt(2022,11,3),"1D", "Q"),
       IRS(dt(2022,12,15),"1D", "Q"),
   )
   curvature_constraints = [
       (Fly(m1, m2, m3), (curve,), {}),
       (Fly(m2, m3, m4), (curve,), {}),
       (Fly(m3, m4, m5), (curve,), {}),
       (Fly(m4, m5, m6), (curve,), {}),
       (Fly(m5, m6, m7), (curve,), {}),
       (Fly(m6, m7, m8), (curve,), {}),
       (Fly(m7, m8, m9), (curve,), {}),
   ]
   instruments = [
       (IRS(dt(2022, 1, 1), "1D", "Q"), (curve,), {}),
       (IRS(dt(2022, 1, 1), "1Y", "Q"), (curve,), {}),
   ]
   s = np.array([0.25, 2.00, 0, 0, 0, 0, 0, 0, 0])
   solver = Solver(
       curves = [curve],
       instruments = instruments + curvature_constraints,
       s = s,
       weights = [1, 1, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
       algorithm = "levenberg_marquardt",
   )
   fig, ax, line = curve.plot("1D")
   plt.show()

This curve has solved exactly and each step-up at each central bank
meeting is priced equally. In the absence of any other information to
the contrary this is a reasonable curve.