.. _api-doc:

**************
API Reference
**************

.. Global imports for iPython are executed in the doc string for add_tenor which is the
   first function indexed alphabetically by automodapi.

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Notation
=========

.. math::

   d \quad \text{or} \quad d_i =& \text{DCF of period} \; (i) \\
   m \quad \text{or} \quad m_i =& \text{Maturity date of period} \; (i) \\
   v(m) =& \text{DF of period payment date,} \; m \\
   N \quad \text{or} \quad N_i =& \text{Notional of period} \; (i) \\
   R =& \text{Fixed rate of period or leg} \\
   z =& \text{Floating period spread} \\
   r(r_i, z) =& \text{Floating rate of period as a function of fixings,} \; (r_i) \\
   C =& \text{Cashflow} \\
   P =& \text{Net present value} \\
   I(m) =& \text{Index ratio applicable at maturity,} \; m \\
   S =& \text{Fixed credit spread} \\
   Q(m) =& \text{Survival probability at maturity,} \; m \\

Defaults
=========

.. automodapi:: rateslib.default
   :no-heading:
   :no-inheritance-diagram:
   :skip: Any
   :skip: plot
   :skip: plot3d
   :skip: datetime
   :skip: Enum
   :skip: read_csv
   :skip: Cal
   :skip: NamedCal
   :skip: Series
   :skip: UnionCal


Scheduling
===========

.. automodapi:: rateslib.scheduling
   :no-heading:
   :no-inheritance-diagram:

Piecewise Polynomial Splines
=============================

Functions
^^^^^^^^^

.. autosummary::
   ~rateslib.splines.bsplev_single
   ~rateslib.splines.bspldnev_single
   ~rateslib.splines.evaluate

Classes
^^^^^^^^

.. autosummary::
   ~rateslib.splines.PPSplineF64
   ~rateslib.splines.PPSplineDual
   ~rateslib.splines.PPSplineDual2


.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    api/rateslib.splines.PPSplineF64.rst
    api/rateslib.splines.PPSplineDual.rst
    api/rateslib.splines.PPSplineDual2.rst
    api/rateslib.splines.bsplev_single.rst
    api/rateslib.splines.bspldnev_single.rst
    api/rateslib.splines.evaluate.rst


Dual (for AD)
==============

.. automodapi:: rateslib.dual
   :no-heading:
   :no-inheritance-diagram:

.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    api/rateslib.dual.Dual.rst
    api/rateslib.dual.Dual2.rst


Curves
=======

.. automodapi:: rateslib.curves
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Class Inheritance Diagram
--------------------------

.. inheritance-diagram:: rateslib.curves.Curve rateslib.curves.LineCurve rateslib.curves.CompositeCurve rateslib.curves.MultiCsaCurve rateslib.curves.ProxyCurve rateslib.curves._BaseCurve rateslib.curves._WithMutability rateslib.curves.CreditImpliedCurve
   :private-bases:
   :parts: 1


FX
===

.. automodapi:: rateslib.fx
   :no-heading:
   :no-inheritance-diagram:

FX Volatility
==============

.. automodapi:: rateslib.fx_volatility
   :inherited-members:
   :no-heading:
   :no-inheritance-diagram:

Periods
========

Link to the :ref:`Periods<periods-doc>` section in the user guide.

.. automodapi:: rateslib.periods
   :no-heading:

Legs
=====

Link to the :ref:`Legs<legs-doc>` section in the user guide.

.. automodapi:: rateslib.legs
   :no-heading:

Instruments
============

.. automodapi:: rateslib.instruments
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Class Inheritance Diagram
--------------------------

.. automod-diagram:: rateslib.instruments
    :parts: 1

Solver
=======

.. automodapi:: rateslib.solver
   :no-heading:
   :skip: Curve
   :skip: ParamSpec
   :skip: MultiCsaCurve
   :skip: NoInput
   :skip: FXRates
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXForwards
   :skip: combinations
   :skip: MultiIndex
   :skip: Series
   :skip: CompositeCurve
   :skip: ProxyCurve
   :skip: concat
   :skip: dual_solve
   :skip: gradient
   :skip: time
   :skip: uuid4
   :skip: PerformanceWarning

Serialization
=============

.. automodapi:: rateslib.serialization
   :inherited-members:
   :no-inheritance-diagram:

Cookbook
=========

Please see :ref:`here for the cookbook index <cookbook-doc>`.
