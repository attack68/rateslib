.. _c-solver-doc:

.. ipython:: python
   :suppress:

   import warnings
   warnings.filterwarnings('always')
   from rateslib.solver import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

***********
Solver
***********

The ``rateslib.solver`` module includes a :class:`~rateslib.solver.Solver` class
which iteratively solves the DFs of :class:`~rateslib.curves.Curve` objects, and
values of :class:`~rateslib.curves.LineCurve` to fit given
calibrating instruments.

This module relies on the utility module :ref:`dual<dual-doc>` for gradient based
optimization.

.. autosummary::
   rateslib.solver.Solver

Calibrating Curves
******************

The above :class:`Curve` was directly specified by known DFs. This is quite an
unusual way of specifying curves in practice. Normally other information,
such as market prices, are provided and the DFs are derived to provide the
best match. When solving DFs from the prices, ``s``, of calibrating instruments,
we must use the :class:`Solver`.

.. ipython:: python
   :okwarning:

   ll_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967
       },
       interpolation="log_linear",
       id="curve",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "Q", curves="curve"),
   ]
   s = np.array([1.0, 1.6, 2.0])
   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = s,
   )
   ll_curve.plot("1D")

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   ll_curve = Curve(
       nodes={dt(2022,1,1): 1.0, dt(2023,1,1): 0.99, dt(2024,1,1): 0.965, dt(2025,1,1): 0.93},
       interpolation="log_linear",
       id="curve"
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "Q", curves="curve"),
   ]
   s = np.array([1.0, 1.6, 2.0])
   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = s,
   )
   fig, ax, line = ll_curve.plot("1D")
   plt.show()

The values of the ``solver.s`` can be updated and the curves can be redetermined

.. ipython:: python

   print(instruments[1].rate(ll_curve).real)
   solver.s[1] = 1.5
   solver.iterate()
   print(instruments[1].rate(ll_curve).real)

To utilise :ref:`spline interpolation<splines-doc>` we must specify the knots,
in standard form. Spline interpolation will
be implemented after the first knot date, which can be set to the initial node
date of the curve for full spline interpolation. Spline curves must be solved if their
spline coefficients are not provided at initialisation. In the below case, spline
interpolation is applied after the first year and prior to that log-linear
interpolation is used.

.. ipython:: python
   :okwarning:

   spline_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1)],
       id="curve",
   )
   solver = Solver(
       curves = [spline_curve],
       instruments = instruments,
       s = s,
   )
   ll_curve.plot("1D", comparators=[spline_curve])

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   ll_curve = Curve(
       nodes={dt(2022,1,1): 1.0, dt(2023,1,1): 0.99, dt(2024,1,1): 0.965, dt(2025,1,1): 0.93},
       interpolation="log_linear",
       id="curve",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "Q", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "Q", curves="curve"),
   ]
   s = np.array([1.0, 1.5, 2.0])
   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = s,
   )
   spline_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1)],
       id="curve",
   )
   solver = Solver(
       curves = [spline_curve],
       instruments = instruments,
       s = s,
   )
   fig, ax, lines = ll_curve.plot("1D", comparators=[spline_curve])
   plt.show()
