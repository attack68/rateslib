.. _curves-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

********************
Constructing Curves
********************

*Rateslib* has six different interest rate *Curve* classes. Three of these are fundamental base
*Curves* of different types and for different purposes. Three are objects which are
constructed via references to other curves to allow certain combinations. *Rateslib* also
has one type of *Smile* for *FX volatility*.

The three fundamental curve classes are:

.. autosummary::
   rateslib.curves.Curve
   rateslib.curves.LineCurve
   rateslib.curves.IndexCurve

The remaining, more complex, combination classes are:

.. autosummary::
   rateslib.curves.CompositeCurve
   rateslib.curves.ProxyCurve
   rateslib.curves.MultiCsaCurve

In *rateslib* **defining curves** and then **solving them with calibrating
instruments** are two separate processes. This provides maximal flexibility whilst
providing a process that is fully generalised and consistent throughout.

.. warning::
   *Rateslib* **does not bootstrap**. Bootstrapping is an analytical process that
   determines curve parameters sequentially and exactly by solving a series of
   equations for a well defined set of parameters and instruments. All curves that
   can be bootstrapped can also be solved by a numerical optimisation routine.

The following pages describe these two processes.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    c_curves.rst
    c_solver.rst
    c_fx_smile.rst

The below provides a basic introduction, exemplifying how a SOFR curve,
which might otherwise be bootstrapped, can be constructed in *rateslib* using the
generalised process of:

- **defining the** ``Curves``,
- **defining the calibrating** ``Instruments``,
- **combining** in the :class:`~rateslib.solver.Solver` with target market prices.

.. ipython:: python

   from rateslib.curves import Curve
   from rateslib.solver import Solver
   from rateslib.instruments import IRS
   from rateslib import dt

   sofr = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 2, 1): 1.0,  # 1M node
           dt(2022, 3, 1): 1.0,  # 2M node
           dt(2022, 4, 1): 1.0,  # 3M node
           dt(2022, 7, 1): 1.0,  # 6M node
           dt(2023, 1, 1): 1.0,  # 1Y node
           dt(2024, 1, 1): 1.0,  # 2Y node
       },
       id="sofr",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "2M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "3M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "6M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "1Y", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="sofr"),
   ]
   rates = [2.4, 2.55, 2.7, 3.1, 3.6, 3.9]
   solver = Solver(
       curves=[sofr],
       instruments=instruments,
       s=rates,
   )
   sofr.nodes
   sofr.plot("1d", labels=["example sofr o/n curve"])

.. plot::

   from rateslib.curves import Curve
   from rateslib.solver import Solver
   from rateslib.instruments import IRS
   from rateslib import dt

   sofr = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 2, 1): 1.0,  # 1M node
           dt(2022, 3, 1): 1.0,  # 2M node
           dt(2022, 4, 1): 1.0,  # 3M node
           dt(2022, 7, 1): 1.0,  # 6M node
           dt(2023, 1, 1): 1.0,  # 1Y node
           dt(2024, 1, 1): 1.0,  # 2Y node
       },
       id="sofr",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "2M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "3M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "6M", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "1Y", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="sofr"),
   ]
   rates = [2.4, 2.55, 2.7, 3.1, 3.6, 3.9]
   solver = Solver(
       curves=[sofr],
       instruments=instruments,
       s=rates,
   )
   fig, ax, line = sofr.plot("1D", labels=["example sofr o/n curve"])
   plt.show()
   plt.close()
