.. _spec-inr-mum:

*************
MUM Calendar
*************

Mumbai business days defined only by the following (ad hoc union government declared
days not yet included):

- Republic Day
- Good Friday
- Ambedkar Jayanti
- May Day
- Independence Day
- Gandhi Jayanti
- Christmas Day

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   get_calendar("mum").holidays
