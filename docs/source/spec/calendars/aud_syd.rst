.. _spec-aud-syd:

*************
SYD Calendar
*************

This defines RITS business days: https://www.rba.gov.au/payments-and-infrastructure/rits/business-hours.html_

Including NSW specific holidays see :ref:`nsw <spec-aud-nsw>`.

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   get_calendar("syd").holidays
