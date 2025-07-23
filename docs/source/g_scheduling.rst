.. _scheduling-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import FXRates, FXForwards
   from rateslib.curves import Curve
   from rateslib.instruments import Value
   from rateslib.solver import Solver
   from datetime import datetime as dt
   import math

************
Scheduling
************

The ``rateslib.scheduling`` module provides calendar, date manipulation and scheduling functionality.

All of *rateslib's* scheduling objects have their base implementation written in Rust, for which
the lower level documentation is available at :rust:`scheduling`.

.. toctree::
    :maxdepth: 2
    :caption: Contents:
    :titlesonly:

    u_calendars.rst
    u_scheduling.rst
    u_dcfs.rst
