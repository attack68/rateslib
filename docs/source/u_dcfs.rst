.. _dcf-doc:

.. ipython:: python
   :suppress:

   from rateslib.scheduling import *
   from datetime import datetime as dt

**************************
Day count fractions (DCFs)
**************************

This module also contains a :meth:`~rateslib.scheduling.dcf` method for calculating
day count fractions.
Review the API documentation for specific calculation details. Current DCF conventions
available are listed below:

.. ipython:: python

   from rateslib.scheduling.dcfs import _DCF
   print(_DCF.keys())
