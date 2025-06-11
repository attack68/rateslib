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

.. inheritance-diagram:: rateslib.curves.Curve rateslib.curves.LineCurve rateslib.curves.CompositeCurve rateslib.curves.MultiCsaCurve rateslib.curves.ProxyCurve rateslib.curves._BaseCurve rateslib.curves._WithMutation
   :private-bases:
   :parts: 1

The ``rateslib.curves`` module allows flexible and powerful curve objects to be created, which
can then, also, be calibrated by a :class:`~rateslib.solver.Solver` and market instruments.

*Rateslib* makes a distinction between two fundamentally different
:class:`~rateslib.curves._CurveType`. One is **values** based and one is **discount factor (DF)**
based.

The fundamental object is the :class:`~rateslib.curves._BaseCurve` abstract base class. All
curve types in *rateslib* inherit this class and provide its methods and operations. All that is
required for an object to inherit a :class:`~rateslib.curves._BaseCurve` is that it provides
a :meth:`~rateslib.curves._BaseCurve.__getitem__` method.

The methods available to *any* :class:`~rateslib.curves._BaseCurve`, based on its
specified binary :class:`~rateslib.curves._CurveType` classification are described below:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Operation
     - **_CurveType.values**
     - **_CurveType.dfs**
   * - `__getitem__(date)`
     - Must return rates.
     - Must return DFs (or survival probabilities implying hazard rates).
   * - :meth:`~rateslib.curves._BaseCurve.rate`
     - Returns just the rate associated with ``effective``.
     - Returns rates with more features; can imply rates of different tenors or add ``float_spread``
       under different compounding methods, derived from DFs.
   * - :meth:`~rateslib.curves._BaseCurve.plot`
     - Creates a *(date, rate)* plot.
     - Creates a *(date, rate)* plot with the additional features as above.
   * - :meth:`~rateslib.curves._BaseCurve.shift`
     - Add a ``spread`` to the *rate*.
     - Add a ``spread`` to the overnight rates implied by the curve.
   * - :meth:`~rateslib.curves._BaseCurve.roll`
     - Translate the rate space in time.
     - Translate the rate space in time.
   * - :meth:`~rateslib.curves._BaseCurve.translate`
     - Translate **only** the initial node date forward in time.
     - Translate **only** the initial node date forward in time.
   * - :meth:`~rateslib.curves._BaseCurve.index_value`
     - *Not available*.
     - Returns index values provided the :class:`~rateslib.curves._CurveMeta` contains an
       ``index_base`` value.
   * - :meth:`~rateslib.curves._BaseCurve.plot_index`
     - *Not available*.
     - Creates a *(date, index_value)* plot provided the above requirements.

The **two main** user curve classes are listed below:

.. autosummary::
   rateslib.curves.Curve
   rateslib.curves.LineCurve


Introduction
************

To create a simple curve, with localised interpolation, minimal configuration is
required, only the ``nodes`` are required.

.. ipython:: python
   :okwarning:

   from rateslib import dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,  # <- initial DF (/survival probability) should always be 1.0
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

The initial node date for either curve type is important because it is implied
to be the date of the construction of the curve (i.e. today's date).
When a :class:`~rateslib.curves.Curve` acts as a discount curve any net present
values (NPVs) might assume other features
from this initial node, e.g. the regular settlement date of securities.
This is the also the reason the initial discount factor should also
be exactly 1.0 on a :class:`~rateslib.curves.Curve`.

The only exception to this is when building a curve used to forecast values, such as *index values*
and inflation prints, it may be practical to start the curve using the most recent
inflation print which is usually assigned to the start of the month,
thus this may be before *today*.

Get Item
--------

As mentioned, any :class:`~rateslib.curves._BaseCurve` type has a
:meth:`~rateslib.curves._BaseCurve.__getitem__` method appropriate to its
:class:`~rateslib.curves._CurveType`.

.. note::

   DFs (and values) before the curve's initial node date return
   **zero**, in order to value historical cashflows at zero.

.. warning::

   DFs and values after the curve's final node date will return a value that is
   an **extrapolation**. This may not be a sensible or well constrained value depending upon the
   interpolation method.

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

Visualization methods, of rates, are also available via
:meth:`_BaseCurve.plot()<rateslib.curves._BaseCurve.plot>`. This allows the easy
inspection of curves directly. Below we demonstrate a plot highlighting the
differences between our parametrised :class:`~rateslib.curves.Curve`
and :class:`~rateslib.curves.LineCurve`.

.. ipython:: python
   :okwarning:

   curve.plot(
       "1D",
       comparators=[linecurve],
       labels=["Curve", "LineCurve"]
   )

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from rateslib import dt
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

*Rateslib* treats curve interpolation in two ways;

- it allows a :class:`~rateslib.curves._CurveSpline` with defined **knot sequence** for
  interpolating ``nodes`` with a cubic :class:`PPSpline <rateslib.splines.PPSplineF64>`.
- it allows **local interpolation** which uses some function to derive a result from only the
  immediately neighbouring ``nodes`` to the input *date*.

If a **spline** is specified and *date* falls between its **knots** it will take precedence.
Otherwise, if the *date* falls outside of the **knots** or if a spline is not specified then
**local interpolation** functions are used.

The available local interpolation options are described in the documentation for each curve class,
and also in supplementary materials, generally they allow the commonly used
*"linear"*, *"log_linear"*, *"flat_forward"* varieties as well as others.

``interpolation`` can also be specified as a **user defined function**, which allows more
flexibility than just local interpolation if required. See
class documentation for required argument signature.

.. ipython:: python

   def linear_with_randomness(date, curve):
       from rateslib.curves.interpolation import index_left
       from random import random
       i = index_left(curve.nodes.keys, curve.nodes.n, date)
       x_1, x_2 = curve.nodes.keys[i], curve.nodes.keys[i + 1]
       y_1, y_2 = curve.nodes.values[i], curve.nodes.values[i + 1]
       return (random() -0.5) * 0.05 + y_1 + (y_2 - y_1) * (date - x_1) / (x_2 - x_1)

   random_lc = LineCurve(
       nodes={
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       },
       interpolation=linear_with_randomness,
   )
   random_lc.plot("1D", comparators=[linecurve], labels=["Random", "LineCurve"])

.. plot::

   from rateslib.curves import *
   import matplotlib.pyplot as plt
   from rateslib import dt
   import numpy as np

   linecurve = LineCurve(
       nodes={
           dt(2022, 1, 1): 0.975,  # <- initial value is general
           dt(2023, 1, 1): 1.10,
           dt(2024, 1, 1): 1.22,
           dt(2025, 1, 1): 1.14,
           dt(2026, 1, 1): 1.03,
           dt(2027, 1, 1): 1.03,
       },
       interpolation="linear",
   )

   def linear_with_randomness(date, curve):
       from rateslib.curves.interpolation import index_left
       from random import random
       i = index_left(curve.nodes.keys, curve.nodes.n, date)
       x_1, x_2 = curve.nodes.keys[i], curve.nodes.keys[i + 1]
       y_1, y_2 = curve.nodes.values[i], curve.nodes.values[i + 1]
       return (random() -0.5) * 0.05 + y_1 + (y_2 - y_1) * (date - x_1) / (x_2 - x_1)

   random_lc = LineCurve(
       nodes={
           dt(2022,1,1): 0.975,  # <- initial value is general
           dt(2023,1,1): 1.10,
           dt(2024,1,1): 1.22,
           dt(2025,1,1): 1.14,
           dt(2026,1,1): 1.03,
           dt(2027,1,1): 1.03,
       },
       interpolation=linear_with_randomness,
   )
   fig, ax, line = random_lc.plot("1D", comparators=[linecurve], labels=["Random", "LineCurve"])
   plt.show()
   plt.close()


Spline Interpolation
---------------------

**Splines** can be automatically created by adding ``interpolation="spline"`` to the initialization
of a curve. This will define a default **knot sequence** that encompasses the whole of the
``nodes`` domain. **DF** based curves' splines will interpolate over the logarithm of DFs, whilst
**values** based curves' splines interpolate directly over those values.

Greater customisation is achieved by directly supplying the **knot sequence** as the ``t``
argument to a curve initialization. This is a list of datetimes and follows the
appropriate mathematical convention for such sequences (see :ref:`pp splines<splines-doc>`).

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
   from rateslib import dt
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


.. _c-curves-ibor-rfr:

IBOR or RFR
************

The different :ref:`Instruments<instruments-toc-doc>` in *rateslib* may require
different interest rate index types, be it IBOR or RFR based. These are
fundamentally different and require care dependent on
which curve type: :class:`~rateslib.curves.Curve` or
:class:`~rateslib.curves.LineCurve` is used. This is also similar to ``fixing`` input
for :class:`~rateslib.periods.FloatPeriod` (see :ref:`here<float fixings>`).

.. list-table::
   :widths: 10 45 45
   :header-rows: 1

   * - Curve Type
     - RFR Based
     - IBOR Based
   * - :class:`~rateslib.curves.Curve`
     - DFs are value date based. For an RFR rate applicable between a start and end
       date, the start and end date DFs will reflect this rate, regardless of the
       publication timeframe of the rate.
     - DFs are value date based. For an IBOR rate applicable between a start and end
       date, the start and end date DFs will reflect this rate, regardless of the
       publication timeframe of the rate.
   * - :class:`~rateslib.curves.LineCurve`
     - Rates are labelled by **reference value date**, **not** publication date.
     - Rates are labelled by **publication date**, **not** reference value date.

Since DF based curves behave similarly for each index type we will give an example
of constructing an :class:`~rateslib.instruments.IRS` under the different methods.

For an RFR curve the ``nodes`` values are by reference date. The 3.0% value which
is applicable between the reference date of 2nd Jan '22 and end date 3rd Jan '22,
is indexed according to the 2nd Jan '22.

.. ipython:: python

   rfr_curve = LineCurve(
       nodes={
           dt(2022, 1, 1): 2.0,
           dt(2022, 1, 2): 3.0,
           dt(2022, 1, 3): 4.0
       }
   )
   irs = IRS(
       dt(2022, 1, 2),
       "1d",
       "A",
       leg2_fixing_method="rfr_payment_delay"
   )
   irs.rate(rfr_curve)

For an IBOR curve the ``nodes`` values are by publication date. The curve below has a
lag of 2 business days. and the publication on 1st Jan '22 is applicable to the
reference value date of 3rd Jan.

.. ipython:: python

   ibor_curve = LineCurve(
       nodes={
           dt(2022, 1, 1): 2.5,
           dt(2022, 1, 2): 3.5,
           dt(2022, 1, 3): 4.5
       }
   )
   irs = IRS(
       dt(2022, 1, 3),
       "3m",
       "A",
       leg2_fixing_method="ibor",
       leg2_method_param=2
   )
   irs.rate(ibor_curve)


Mutable Pricing Objects
*************************

The only curves with parameters that are mutated and solved by a :class:`~rateslib.solver.Solver`
are :class:`~rateslib.curves.Curve` and :class:`~rateslib.curves.LineCurve`. These are
classed as *Pricing Objects*.

These curves inherit the :class:`~rateslib.curves._WithMutation` mixin.

Pricing Containers
********************

Other objects that are available, that are constructed via manipulations of the base *Pricing
Objects* (or other *Pricing Containers*) are the so called *Pricing Containers*.

The main user curve classes are listed below:

.. autosummary::
   rateslib.curves.CompositeCurve
   rateslib.curves.MultiCsaCurve
   rateslib.curves.ProxyCurve

These objects allow complex curve features and scenarios to be modelled in a recognisable and
easily parametrised format.

The following *Pricing Containers* are also created as the result of certain operations:

.. autosummary::
   rateslib.curves._ShiftedCurve
   rateslib.curves._RolledCurve
   rateslib.curves._TranslatedCurve
