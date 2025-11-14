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


Defaults
=========

These objects are used to set values for missing parameters globally.

.. automodapi:: rateslib.default
   :no-heading:
   :no-inheritance-diagram:

Enums
======

Objects to define parameter settings across different objects throughout the libary.

.. automodapi:: rateslib.enums
   :no-heading:
   :no-inheritance-diagram:

Scheduling
===========

.. automodapi:: rateslib.scheduling
   :no-heading:
   :no-inheritance-diagram:

Piecewise Polynomial Splines
=============================

.. automodapi:: rateslib.splines
   :no-heading:
   :no-inheritance-diagram:


Dual (for AD)
==============

.. automodapi:: rateslib.dual
   :no-heading:
   :no-inheritance-diagram:


Curves
=======

.. automodapi:: rateslib.curves
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: rateslib.curves.Curve rateslib.curves.LineCurve rateslib.curves.CompositeCurve rateslib.curves.MultiCsaCurve rateslib.curves.ProxyCurve rateslib.curves._BaseCurve rateslib.curves._WithMutability rateslib.curves.CreditImpliedCurve rateslib.curves.TranslatedCurve rateslib.curves.RolledCurve rateslib.curves.ShiftedCurve
   :private-bases:
   :parts: 1


Fixings
========

.. automodapi:: rateslib.data.loader
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

.. automodapi:: rateslib.data.fixings
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

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

Protocols
^^^^^^^^^

*Period* protocols establish common functionality and methods that all *Periods* share consistently.

.. automodapi:: rateslib.periods.components.protocols
   :no-inheritance-diagram:
   :headings: "^-"
   :inherited-members:
   :no-heading:
   :skip: _BasePeriod
   :skip: _BasePeriodStatic

Parameters
^^^^^^^^^^

*Period* parameters define containers for input values used in the construction of functionality.

.. automodapi:: rateslib.periods.components.parameters
   :no-inheritance-diagram:
   :headings: "^-"
   :inherited-members:
   :no-heading:
   :skip: _init_or_none_IndexParams
   :skip: _init_or_none_NonDeliverableParams
   :skip: _init_SettlementParams_with_fx_pair
   :skip: _init_FloatRateParams
   :skip: _init_MtmParams

Objects
^^^^^^^^

.. automodapi:: rateslib.periods.components
   :no-inheritance-diagram:
   :headings: "^-"
   :inherited-members:
   :no-heading:


Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: rateslib.periods.components.protocols._WithNPV rateslib.periods.components.protocols._WithNPVStatic rateslib.periods.components.protocols._WithIndexingStatic rateslib.periods.components.protocols._WithNonDeliverableStatic rateslib.periods.components.BasePeriod rateslib.periods.components.FloatPeriod rateslib.periods.components.Cashflow rateslib.periods.components.FixedPeriod
   :private-bases:
   :parts: 1


Legs
=====

Link to the :ref:`Legs<legs-doc>` section in the user guide.

.. automodapi:: rateslib.legs
   :no-heading:


.. Protocols
    ^^^^^^^^^

    .. automodapi:: rateslib.legs.components.protocols
       :no-inheritance-diagram:
       :headings: "^-"
       :inherited-members:
       :no-heading:
       :include: _WithNPV
       :include: _WithNPVStatic
       :include: _WithIndexingStatic
       :include: _WithNonDeliverableStatic

    Parameters
    ^^^^^^^^^^

    .. automodapi:: rateslib.legs.components.parameters
       :no-inheritance-diagram:
       :headings: "^-"
       :inherited-members:
       :no-heading:
       :skip: _init_or_none_IndexParams
       :skip: _init_or_none_NonDeliverableParams
       :skip: _init_SettlementParams_with_fx_pair
       :skip: _init_FloatRateParams
       :skip: _init_MtmParams

Objects
^^^^^^^^

.. automodapi:: rateslib.legs.components
   :no-inheritance-diagram:
   :headings: "^-"
   :inherited-members:
   :no-heading:
   :include: FixedLeg
   :include: FloatLeg
   :include: Amortization

Instruments
============

.. automodapi:: rateslib.instruments
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Objects
^^^^^^^^

.. automodapi:: rateslib.instruments.components
   :no-inheritance-diagram:
   :headings: "^-"
   :inherited-members:
   :no-heading:
   :include: IRS
   :include: NDF
   :include: XCS


Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
==============

.. automodapi:: rateslib.serialization
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:


Cookbook
=========

Please see :ref:`here for the cookbook index <cookbook-doc>`.
