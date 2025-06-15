.. _base-curve-doc:

Building Custom Curves (Nelson-Seigel)
****************************************

The *rateslib* curves objects are structured in such a way that it is
strightforward to create new objects useable throughout the library, provided they implement
the necessary abstract base objects.

This example will construct a parametric **Nelson-Seigel Curve**, whose continuously
compounded zero rate, :math:`r`, at time, :math:`T`, is given by the following
equation of **four** parameters:

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
   from rateslib.curves import _BaseCurve, _CurveType, _WithOperations, _CurveNodes

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
However, it doesn't do much without implementing the ``__getitem__`` method.

.. ipython:: python

   curve = NelsonSeigelCurve(dt(2000, 1, 1), dt(2010, 1, 1), 0.025, 0.0, 0.0, 0.5)

The __getitem__ method
-----------------------

If the requested date is prior to the curve: return zero as usual. If the date is the
same as the the initial node date: return one, else use continuously compounded rates to
derive a discount factor.

.. ipython:: python

   def __getitem__(self, date):
       if date < self.nodes.initial:
           return 0.0
       elif date == self.nodes.initial:
           return 1.0
       b0, b1, b2, l0 = self._params
       T = dcf(self.nodes.initial, date, convention=self.meta.convention, calendar=self.meta.calendar)
       a1 = l0 * (1 - dual_exp(-T / l0)) / T
       a2 =  a1 - dual_exp(-T / l0)
       r = b0 + a1 * b1 + a2 * b2
       return dual_exp(-T * r)

.. ipython:: python
   :suppress:

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
           if date < self.nodes.initial:
               return 0.0
           elif date == self.nodes.initial:
               return 1.0
           b0, b1, b2, l0 = self._params
           T = dcf(self.nodes.initial, date, convention=self.meta.convention, calendar=self.meta.calendar)
           a1 = l0 * (1 - dual_exp(-T / l0)) / T
           a2 =  a1 - dual_exp(-T / l0)
           r = b0 + a1 * b1 + a2 * b2
           return dual_exp(-T * r)

Once this method is added to the class and the discount factors are available,
all of the provided methods are also available. This means the following are all
automatically functional:

.. ipython:: python

   ns_curve = NelsonSeigelCurve(dt(2000, 1, 1), dt(2010, 1, 1), 0.03, -0.01, 0.01, 0.75)
   ns_curve.plot("1b", comparators=[ns_curve.shift(100), ns_curve.roll("6m")])

.. plot::

   from rateslib import *
   from rateslib.curves import _BaseCurve, _CurveType, _WithOperations, _CurveNodes
   import matplotlib.pyplot as plt

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
           if date < self.nodes.initial:
               return 0.0
           elif date == self.nodes.initial:
               return 1.0

           b0, b1, b2, l0 = self._params
           T = dcf(self.nodes.initial, date, convention=self.meta.convention, calendar=self.meta.calendar)
           a1 = l0 * (1 - dual_exp(-T / l0)) / T
           a2 =  a1 - dual_exp(-T / l0)
           r = b0 + a1 * b1 + a2 * b2
           return dual_exp(-T * r)

   ns_curve = NelsonSeigelCurve(dt(2000, 1, 1), dt(2010, 1, 1), 0.03, -0.01, 0.01, 0.75)
   fix, ax, lines = ns_curve.plot("1b", comparators=[ns_curve.shift(100), ns_curve.roll("6m")])
   plt.show()
   plt.close()

Mutatbility, the ``Solver`` and Risk
--------------------------------------

In order to allow this curve to be calibrated by a :class:`~rateslib.solver.Solver`,
we need to add some elements that allows the :class:`~rateslib.solver.Solver` to interact
with it. We will also set the `NelsonSeigelCurve` to inherit
:class:`~rateslib.curves._WithMutability`.

Firstly, we can add the ``getter`` methods (NumPy is needed for this).
Make sure that ``_ini_solve = 0`` is added as a property to the class.

.. ipython:: python

   _ini_solve = 0

   def _get_node_vector(self):
       return np.array(self._params)

   def _get_node_vars(self):
       return tuple(f"{self._id}{i}" for i in range(self._ini_solve, self._n))

The ``setter`` method that the :class:`~rateslib.solver.Solver` needs is slightly
more complicated. It requires state management, which we can easily add. The additional
methods are shown below.

.. ipython:: python

   from rateslib.curves import _WithMutability
   from rateslib.mutability import _new_state_post
   from rateslib.dual import set_order_convert
   from rateslib.dual.utils import _dual_float

   @_new_state_post
   def _set_node_vector(self, vector, ad):
        if ad == 0:
            self._params = tuple(_dual_float(_) for _ in vector)
        elif ad == 1:
            self._params = tuple(
                Dual(_dual_float(_), [f"{self._id}{i}"], []) for i, _ in enumerate(vector)
            )
        else: # ad == 2
            self._params = tuple(
                Dual2(_dual_float(_), [f"{self._id}{i}"], [], []) for i, _ in enumerate(vector)
            )

   def _set_ad_order(self, order):
       if self.ad == order:
           return None
       else:
           self._params = tuple(
               set_order_convert(_, order, [f"{self._id}{i}"]) for i, _ in enumerate(self._params)
           )

Adding these elements yields the final code class:

.. ipython:: python

   class NelsonSeigelCurve(_WithMutability, _WithOperations, _BaseCurve):
       _ini_solve = 0
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
       def __getitem__(self, date):
           if date < self.nodes.initial:
               return 0.0
           elif date == self.nodes.initial:
               return 1.0
           b0, b1, b2, l0 = self._params
           T = dcf(self.nodes.initial, date, convention=self.meta.convention, calendar=self.meta.calendar)
           a1 = l0 * (1 - dual_exp(-T / l0)) / T
           a2 =  a1 - dual_exp(-T / l0)
           r = b0 + a1 * b1 + a2 * b2
           return dual_exp(-T * r)
       def _get_node_vector(self):
           return np.array(self._params)
       def _get_node_vars(self):
           return tuple(f"{self._id}{i}" for i in range(self._ini_solve, self._n))
       def _set_node_vector(self, vector, ad):
           if ad == 0:
               self._params = tuple(_dual_float(_) for _ in vector)
           elif ad == 1:
               self._params = tuple(
                    Dual(_dual_float(_), [f"{self._id}{i}"], []) for i, _ in enumerate(vector)
               )
           else: # ad == 2
               self._params = tuple(
                   Dual2(_dual_float(_), [f"{self._id}{i}"], [], []) for i, _ in enumerate(vector)
               )
       def _set_ad_order(self, order):
           if self.ad == order:
               return None
           else:
               self._params = tuple(
                   set_order_convert(_, order, [f"{self._id}{i}"]) for i, _ in enumerate(self._params)
               )
