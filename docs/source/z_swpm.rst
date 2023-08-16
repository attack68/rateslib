.. _swpm-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame

***********
Example
***********

Replicating a SOFR Curve from Bloomberg's SWPM
**********************************************

At a point in time on Wed 16th Aug 2023 loading the SWPM function in Bloomberg
presented the following default SOFR curve data:

.. image:: _static/sofr_swpm_1.PNG
  :alt: SWPM SOFR Curve

.. ipython:: python

   data = DataFrame({
       "Term": ["1W", "2W", "3W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "12M", "18M", "2Y", "3Y", "4Y"],
       "Rate": [5.30190, 5.30470, 5.30732, 5.31038, 5.33912, 5.37200, 5.40000, 5.41712, 5.42800, 5.43218, 5.42928, 5.41930, 5.40600, 5.38300, 5.35550, 5.04775, 4.79988, 4.44106, 4.22650],
   })
   data
