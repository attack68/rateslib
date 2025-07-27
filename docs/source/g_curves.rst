.. _curves-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

****************************
Constructing Pricing Objects
****************************

Curves
------

*Rateslib* has **two** fundamental curve classes:

.. autosummary::
   rateslib.curves.Curve
   rateslib.curves.LineCurve

There are also additional, more complex objects, which serve as containers which composite
multiple objects.

.. autosummary::
   rateslib.curves.CompositeCurve
   rateslib.curves.ProxyCurve
   rateslib.curves.MultiCsaCurve
   rateslib.curves.CreditImpliedCurve

And there are other containers which facilitate different types of curve *operations*.

.. autosummary::
   rateslib.curves.RolledCurve
   rateslib.curves.ShiftCurve
   rateslib.curves.TranslateCurve

It is also possible for a user to construct **custom curve classes** and interact them with
all of the established tools, in particular for pricing instruments and obtaining risk
sensitivities. For an example of this see the cookbook article on
`Building Custom Curves (Nelosn-Siegel) <z_basecurve.html>`_.

Curves are required for pricing all *Instruments* in *rateslib*.

Smiles & Surfaces
-----------------

For FX volatility pricing *rateslib* has two fundamental models:

.. autosummary::
   rateslib.fx_volatility.FXDeltaVolSmile
   rateslib.fx_volatility.FXSabrSmile

These cross-sectional elements are combined into a generalist surface which interpolates volatility
in the objects:

.. autosummary::
   rateslib.fx_volatility.FXDeltaVolSurface
   rateslib.fx_volatility.FXSabrSurface

Solver
-------

In *rateslib* **defining pricing objects** and then **solving them with calibrating instruments**
are two separate processes. This provides maximal flexibility whilst
providing a process that is fully generalised and consistent throughout.

.. note::

   *Rateslib* uses a global optimizer and weighted least squares object function to calibrate
   pricing objects.

   *Rateslib* **does not bootstrap**. Bootstrapping is an analytical process that
   determines pricing object parameters sequentially and exactly by solving a series of
   equations for a well defined set of parameters and instruments.

   Any bootstrapped object can be solved, also, by a global optimization process. Many
   global optimization solutions cannot be determined by a bootstrap process.

The following pages give further details to these summaries and code examples.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    c_curves.rst
    c_solver.rst
    c_fx_smile.rst
