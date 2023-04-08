.. _combinations-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**************************************
Utilities and Instrument Combinations
**************************************

Combinations allow common interbank traded structures to be
created, and :class:`~rateslib.instruments.Value` is a null object typically
used to directly parametrize a :ref:`Curve<curves-doc>` via a
:class:`~rateslib.solver.Solver`

.. autosummary::
   rateslib.instruments.Value
   rateslib.instruments.Spread
   rateslib.instruments.Fly
