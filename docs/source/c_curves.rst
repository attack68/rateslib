.. _c-curves-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

***********
Curves
***********

The ``rateslib.curves`` module allows a :class:`~rateslib.curves.Curve` or
:class:`~rateslib.curves.LineCurve` class
to be defined with parameters.
Both of these create curve objects but are inherently different in what they
represent and how they operate.

This module relies on the ultility modules :ref:`splines<splines-doc>`
and :ref:`dual<dual-doc>`.

.. autosummary::
   rateslib.curves.Curve
   rateslib.curves.LineCurve
   rateslib.curves.interpolate
   rateslib.curves.index_left

Each curve type has ``rate``, ``plot``, ``shift``, ``roll`` and ``translate`` methods.

.. autosummary::
   rateslib.curves.Curve.rate
   rateslib.curves.Curve.plot
   rateslib.curves.Curve.shift
   rateslib.curves.Curve.roll
   rateslib.curves.Curve.translate
   rateslib.curves.LineCurve.rate
   rateslib.curves.LineCurve.plot
   rateslib.curves.LineCurve.shift
   rateslib.curves.LineCurve.roll
   rateslib.curves.LineCurve.translate

The main parameter that must be supplied to either type of curve is its ``nodes``. This
provides the curve with its degrees of freedom and represents a dict indexed by
datetimes, each with a given value. In the case of a :class:`~rateslib.curves.Curve`
these
values are discount factors (DFs), and in the case of
a :class:`~rateslib.curves.LineCurve`
these are specific values, usually rates associated with that curve.

Curve
*******

A :class:`~rateslib.curves.Curve` can only be used for **interest rates**.
It is a more specialised
object because of the way it is **defined by discount factors (DFs)**. These DFs
maintain an inherent interpolation technique, which is often log-linear or log-cubic
spline. These are generally the most efficient
type of curve, and most easily parametrised, when working with compounded RFR rates.
The initial node on a :class:`~rateslib.curves.Curve` should always have value 1.0,
and it will not
be varied by a :class:`~rateslib.solver.Solver`. :class:`~rateslib.curves.Curve` s must
be used with
:class:`~rateslib.fx.FXForwards` since FX forwards calculation rely on the existence
of DFs.

LineCurves
***********

A :class:`~rateslib.curves.LineCurve` is a more general object which can be
used to represent other forms of **datetime indexed values**. The values maintain
interpolation
techniques where the most common are likely to be linear and splines. These are
generally quite inefficient, and more difficult to parametrise, when dealing with RFR
rates, but may be superior when dealing with legacy IBOR rates or inflation etc.
The initial node on a :class:`~rateslib.curves.LineCurve` can take any value and it will
be varied by a :class:`~rateslib.solver.Solver`.

Introduction
************

To create a simple curve, with localised interpolation, minimal configuration is
required.

.. ipython:: python
   :okwarning:

   from datetime import datetime as dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,  # <- initial DF should always be 1.0
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
   )

We can also use a similar configuration for a generalised curve constructed from
connecting lines between values.

.. ipython:: python
   :okwarning:

   linecurve = LineCurve(
       nodes={
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       },
       interpolation="linear",
   )

Initial Node Date
-----------------

The initial node date is important because it is is implied to be the date of the
construction of the curve. Any net present values (NPVs) may assume other features
from this initial node, e.g. the regular settlement date of securities or the value of
cashflows on derivatives. This is the reason the initial discount factor should also
be exactly 1.0 on a :class:`~rateslib.curves.Curve`.

Get Item
--------

Curves have a get item method so that valid DFs or values can easily be extracted
under the curve's specified interpolation scheme.

.. note::

   `Curve` DFs (and `LineCurve` values), before the curve's initial node date return
   **zero**, in order to value historical cashflows at zero.

.. warning::

   `Curve` DFs, and `LineCurve` values, after the curve's final node date will
   return a value that is an **extrapolation**.
   This may not be a sensible or well constrained value depending upon the
   interpolation.

.. ipython:: python
   :okwarning:

   curve[dt(2022, 9, 26)]
   curve[dt(1999, 12, 31)]  # <- before the curve initial node date
   curve[dt(2032, 1, 1)]  # <- extrapolated after the curve final node date

.. ipython:: python
   :okwarning:

   linecurve[dt(2022, 9, 26)]
   linecurve[dt(1999, 12, 31)]  # <- before the curve initial node date
   linecurve[dt(2032, 1, 1)]  # <- extrapolated after the curve final node date

Visualization
**************

Visualization methods are inherited by subclassing
:class:`~rateslib.curves.PlotCurve` which provides
the methods :meth:`Curve.plot()<rateslib.curves.Curve.plot>` and
:meth:`LineCurve.plot()<rateslib.curves.LineCurve.plot>`. This allows the easy
inspection of curves directly. Below we demonstrate a plot highlighting the
differences between our parametrised :class:`~rateslib.curves.Curve`
and :class:`~rateslib.curves.LineCurve`.

.. ipython:: python
   :okwarning:

   curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
   )
   linecurve = LineCurve(
       nodes={
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       },
       interpolation="linear",
   )
   # curve_lin = Curve(nodes=curve.nodes, interpolation="linear")
   # curve_zero = Curve(nodes=curve.nodes, interpolation="linear_zero_rate")
   fig, ax, line = curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])
   plt.show()


Interpolation
*************

The available basic local interpolation options are:

- *"linear"*: this is most suitable, and the default,
  for :class:`~rateslib.curves.LineCurve`. Linear interpolation for DF based curves
  usually produces spurious underlying curves.
- *"log_linear"*: this is most suitable, and the default,
  for :class:`~rateslib.curves.Curve`. It produces overnight rates that are constant
  between ``nodes``. This is not usually suitable
  for :class:`~rateslib.curves.LineCurve`.
- *"linear_zero_rate"*:  this is a legacy option for linearly interpolating
  continuously compounded zero rates, and is only suitable for
  :class:`~rateslib.curves.Curve`, but it is not recommended and tends also to
  produce spurious underlying curves.
- *"flat_forward"*: this is only suitable for :class:`~rateslib.curves.LineCurve`, and
  it maintains the previous value between ``nodes``. It will produce a stepped curve
  similar to a :class:`~rateslib.curves.Curve` with *"log_linear"* interpolation.
- *"flat_backward"*: same as above but in reverse.

.. ipython:: python
   :okwarning:

   linecurve.interpolation = "flat_forward"
   curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
   )
   linecurve = LineCurve(
       nodes={
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       },
       interpolation="flat_forward",
   )
   # curve_lin = Curve(nodes=curve.nodes, interpolation="linear")
   # curve_zero = Curve(nodes=curve.nodes, interpolation="linear_zero_rate")
   fig, ax, line = curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])
   plt.show()


``interpolation`` can also be specified as a **user defined function**. It must
have the argument signature *(date, nodes)* where ``nodes`` are passed internally as
those copied from the curve.

.. ipython:: python

   from rateslib.curves import index_left
   def flat_backward(x, nodes):
       """Project the rightmost node value as opposed to leftmost."""
       node_dates = [key for key in nodes.keys()]
       if x < node_dates[0]:
           return 0  # then date is in the past and DF is zero
       l_index = index_left(node_dates, len(node_dates), x)
       return nodes[node_dates[l_index + 1]]

   linecurve.interpolation = flat_backward
   curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

   curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 0.99,
           dt(2024, 1, 1): 0.979,
           dt(2025, 1, 1): 0.967,
           dt(2026, 1, 1): 0.956,
           dt(2027, 1, 1): 0.946,
       },
       interpolation="log_linear",
   )
   linecurve = LineCurve(
       nodes={
           dt(2022, 1, 1): 0.975,  # <- initial value is general
           dt(2023, 1, 1): 1.10,
           dt(2024, 1, 1): 1.22,
           dt(2025, 1, 1): 1.14,
           dt(2026, 1, 1): 1.03,
           dt(2027, 1, 1): 1.03,
       },
       interpolation="flat_forward",
   )

   # curve_lin = Curve(nodes=curve.nodes, interpolation="linear")
   # curve_zero = Curve(nodes=curve.nodes, interpolation="linear_zero_rate")
   def flat_backward(x, nodes):
       node_dates = list(nodes.keys())
       if x < node_dates[0]:
           return 0  # then date is in the past and DF is zero
       l_index = index_left(node_dates, len(node_dates), x)
       return nodes[node_dates[l_index + 1]]

   linecurve.interpolation = flat_backward
   fig, ax, line = curve.plot("1D", comparators=[linecurve], labels=["Curve", "LineCurve"])
   plt.show()

Spline Interpolation
*********************

There is also an option to interpolate with a cubic polynomial spline.

If applying spline interpolation to a :class:`~rateslib.curves.Curve` then it is
applied logarithmically resulting in a log-cubic spline over DFs.

If it is applied to a :class:`~rateslib.curves.LineCurve` then it results in a
standard cubic spline.

In order to instruct this mode of interpolation a **knot sequence** is required
as the ``t`` argument. This is a list of datetimes and follows the
appropriate convention for such sequences (see :ref:`pp splines<splines-doc>`).

Mixed Interpolation
-------------------

Prior to the initial knot in the sequence the local interpolation method
is used. This allows curves to be constructed with a mixed interpolation in two parts of
the curve. This is common practice for interest rate curves usually with a
*log-linear* short end and a *log-cubic spline* longer end.

.. ipython:: python
   :okwarning:

   mixed_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
       t = [dt(2024,1,1), dt(2024,1,1), dt(2024,1,1), dt(2024,1,1),
            dt(2025,1,1),
            dt(2026,1,1),
            dt(2027,1,1), dt(2027,1,1), dt(2027,1,1), dt(2027,1,1)]
   )
   curve.plot("1D", comparators=[mixed_curve], labels=["log-linear", "log-cubic-mix"])

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
   )
   mixed_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,1): 0.967,
           dt(2026,1,1): 0.956,
           dt(2027,1,1): 0.946,
       },
       interpolation="log_linear",
       t = [dt(2024,1,1), dt(2024,1,1), dt(2024,1,1), dt(2024,1,1),
            dt(2025,1,1),
            dt(2026,1,1),
            dt(2027,1,1), dt(2027,1,1), dt(2027,1,1), dt(2027,1,1)]
   )
   fig, ax, line = curve.plot("1D", comparators=[mixed_curve], labels=["log-linear", "log-cubic-mix"])
   plt.show()
