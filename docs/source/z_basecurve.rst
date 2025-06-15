.. _base-curve-doc:

Building Custom Curves (Nelson-Seigel)
****************************************

The *rateslib* curves objects are structured in such a way that it is
strightforward to create new objects useable throughout the library, provided they implement
the necessary abstract base objects.

This example will construct a parametric **Nelson-Seigel Curve**, whose continuously
compounded zero rate at time, :math:`T` is given by the following equation of **four**
parameters:

.. math::

   r(T) = \begin{bmatrix} \beta_0 & \beta_1 & \beta_2 \end{bmatrix} \begin{bmatrix} 1 \\ \lambda (1- e^{-T/ \lambda}) / T \\ \lambda (1- e^{-T/ \lambda})/ T - e^{-T/ \lambda} \end{bmatrix}

_BaseCurve ABCs
-----------------

First we setup the skeletal structure of our custom curve. We will inherit from
:class:`~rateslib.curves._BaseCurve` and setup the necessary abstract base class (ABC) properties.

Some items we know in advance regarding our custom curve:

- it will return discount factors, defining its :class:`~rateslib.curves._CurveType`,
- it has four parameters,
- we must define a ``start`` date (which makes its initial node) and ``end``.

A :class:`~rateslib.curves._BaseCurve` requires implementations of
:class:`~rateslib.curves._BaseCurve.roll`, :class:`~rateslib.curves._BaseCurve.shift`,
:class:`~rateslib.curves._BaseCurve.translate`. We can acquire these automatically by
inheriting the :class:`~rateslib.curves._WithOperations` class.

.. ipython:: python

   from rateslib import *
   from rateslib.curves import _BaseCurve, _CurveType, _WithOperations

   class NelsonSeigelCurve(_WithOperations, _BaseCurve):
       _base_type = _CurveType.dfs
       _id = None
       _meta = None
       _nodes = None
       _ad = None
       _interpolator = None
       _n = 4
       def __init__(self, start, end, beta0, beta1, beta2, lamb):
           self._id = super()._id
           self._meta = super()._meta
           self._ad = 0
           self._nodes = _CurveNodes({start: 0.0, end: 0.0})
           self._params = (beta0, beta1, beta2, lamb)
       def _set_ad_order(self, order):
           raise NotImplementedError()
       def __getitem__(self, date):
           raise NotImplementedError()

This curve can now be initialised without raising any errors relating to *Abstract Base Classes*.
However, it doesn't do much without implementing the ``__getitem`` method.

.. ipython:: python

   curve = NelsonSeigelCurve(dt(2000, 1, 1), dt(2010, 1, 1), 0.025, 0.0, 0.0, 0.5)
