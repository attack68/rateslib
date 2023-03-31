.. _curves-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

************
Curves
************

A misconception in finance is that interest rate curves are defined by par swap rates,
for example, a SOFR curve might be defined by the par SOFR rates:
{1M, 2M, 3M, 6M, 1Y, 2Y, 3Y, etc. }. This is **not** true. Whilst that information *is*
sufficient to create a SOFR curve it is a very narrow subset of the possibilities
permissible for creating curves,
and it exemplifies a misunderstanding of the **generalised process** for constructing
curves. It also excludes sets of other curves such as bond credit curves.

In ``rateslib`` **defining curves** and then **solving them with calibrating
instruments** are two separate processes. There
needs to be weak mathematical association between them (for the solver to be successful)
but it certainly does not need to be as strong as ensuring the nodes on the curve
and the rates supplied are one-to-one equivalent.
The following pages of the guide aim to explain the individual components and their
parameters.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    c_curves.rst
    c_solver.rst

The basic SOFR curve above, exhibiting strong one-to-one association between nodes
and instruments would be created as follows, with the distinct classes as
associated objects.

.. ipython:: python

   from rateslib.curves import Curve
   from rateslib.solver import Solver
   from rateslib.instruments import IRS

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
   from datetime import datetime as dt

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
