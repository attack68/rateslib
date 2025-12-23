.. _combinations-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**************************************
Utilities and Instrument Combinations
**************************************

Combinations
************

Combinations allow common interbank traded structures to be
created inside containers. These can then also be used as an input
to a :class:`~rateslib.solver.Solver` for *Curve*/*Surface* calibration.

.. autosummary::
   rateslib.instruments.Spread
   rateslib.instruments.Fly
   rateslib.instruments.Portfolio

Utilities
*********

The :class:`~rateslib.instruments.Value` and :class:`~rateslib.instruments.FXVolValue` class
serve as null *Instruments*, whose purpose is to directly parametrize a :ref:`Curve<curves-doc>`,
*FXVolSmile* or *FXVolSurface* via a
:class:`~rateslib.solver.Solver`, without *Instrument* construction.

.. autosummary::
   rateslib.instruments.Value
   rateslib.instruments.FXVolValue

As an example the cookbook article :ref:`'Constructing Curves from (CC) Zero Rates'<cookbook-doc>`
shows how to use a *Value* (defined as a
continuously compounded zero rate) to calibrate a *Curve*.