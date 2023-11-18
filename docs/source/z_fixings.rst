.. _cook-fixings-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Index

Working with Fixings
**********************

The *rateslib* `defaults` object lazy loads fixings from CSV files.
These files are stored in the rateslib package files under the directory *'data'*.
Due to static packaging and licencing issues rateslib cannot be distributed
with accurate and upto date RFR or IBOR fixings.

As an example, a small selection of SOFR fixings which are used in some examples
within, *rateslib* documentation are shown below along with the source directory.

.. ipython:: python

   defaults.fixings.directory
   defaults.fixings["usd_rfr"]  # an available alias is 'fixings["sofr"]'

It is possible to overwrite these CSV files (provided the template structure is
maintained) or to set a new directory and place new CSV files there.
Due to lazy loading, this should be done before calling any fixing *Series*, and
all files should be stored in the same folder with the required naming convention.

First we create a new fixing Series for the SEK 3M STIBOR index, and
save it to CSV file in the current working directory.

.. ipython:: python

   df = DataFrame(
       data=[1.21, 2.75],
       index=Index(["02-01-2023", "03-01-2023"], name="reference_date"),
       columns=["rate"]
   )
   print(df)
   df.to_csv("sek_ibor_3m.csv")  # Save the DataFrame and create a CSV file

Next we set the directory of the `defaults.fixings` object and load the fixings.

.. ipython:: python

   import os
   defaults.fixings.directory = os.getcwd()
   defaults.fixings.sek_ibor_3m

Currently the `fixings` object has support only for USD, GBP, EUR, SEK, NOK, CAD, CHF,
with the indexes: RFR, 1M IBOR, 3M IBOR, 6M IBOR and 12M IBOR, but this is easily
extendable via pull request.

These fixings can then be passed to *Instrument* constructors. For STIBOR the
index lag is 2 business days so the fixing for the below *IRS* effective as of
4th January is taken as the value **published on** reference date 2nd Jan 2023.

.. ipython:: python

   irs = IRS(
       effective=dt(2023, 1, 4),
       termination="6M",
       spec="sek_irs3",
       leg2_fixings=defaults.fixings.sek_ibor_3m,
       fixed_rate=2.00,
   )
   curve = Curve({dt(2023, 1, 3): 1.0, dt(2024, 1, 3): 0.98})
   irs.cashflows(curve)
   irs.leg2.fixings_table(curve)
