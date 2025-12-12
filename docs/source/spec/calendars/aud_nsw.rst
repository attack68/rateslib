.. _spec-aud-nsw:

*************
NSW Calendar
*************

This calendar extends the *'SYD'* calendar to include the NSW Bank Holiday (first Monday of August)
and the NSW Labour Day (first Monday of October), which are used with currency settlements.

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   get_calendar("nsw").holidays
