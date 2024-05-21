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

.. note::

   To understand the difference between RFR and IBOR date indexing in *rateslib* see
   :ref:`IBOR or RFR <c-curves-ibor-rfr>`.

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

First we obtain (or create) new fixing for the SEK 3M STIBOR index, and
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
   defaults.fixings["sek_ibor_3m"]

These fixings are entirely user defined in their construction and naming convention. If
an attempt is made to call a fixing series that doesn't exist the user is met with the instructive
error.

.. ipython:: python

   try:
       defaults.fixings["arbitrary_index"]
   except ValueError as e:
       print(e)

Constructing *Instruments* with *fixings*
------------------------------------------

These fixings can then be passed to *Instrument* constructors. For STIBOR the
index lag is 2 business days so the fixing for the below *IRS* effective as of
4th January is taken as the value **published on** the reference date 2nd January.

.. ipython:: python

   irs = IRS(
       effective=dt(2023, 1, 4),
       termination="6M",
       spec="sek_irs3",
       leg2_fixings=defaults.fixings["sek_ibor_3m"],
       fixed_rate=2.00,
   )
   curve = Curve({dt(2023, 1, 3): 1.0, dt(2024, 1, 3): 0.97})
   irs.cashflows(curve)
   irs.leg2.fixings_table(curve)


Debugging with *fixings* on-the-fly
------------------------------------

Using pandas and the *rateslib* calendar methods it is simple to build a series of
*fixings* to match the *Curve* calendar. This is often useful for debugging
historical *FloatPeriods* in the absence of real data.

.. ipython:: python

   from pandas import Series, date_range
   from rateslib.calendars import get_calendar
   fixings = Series(
       data=2.5,
       index=date_range(start=dt(2022, 1, 3), end=dt(2022, 4, 14), freq=get_calendar("nyc")),
   )
   irs = IRS(
       effective=dt(2022, 2, 4),
       termination="6M",
       spec="usd_irs",
       leg2_fixings=fixings,
       fixed_rate=2.00,
   )
   curve = Curve(
       nodes={dt(2022, 4, 15): 1.0, dt(2024, 1, 3): 0.97},
       calendar="nyc",
   )
   irs.cashflows(curve)
   irs.leg2.fixings_table(curve)

Using *fx fixings* in multi-currency *Instruments*
----------------------------------------------------

:class:`~rateslib.instruments.XCS` typically require MTM payments based on FX fixings. However,
the first FX fixing is usually agreed at trade time as the prevailing FX rate at the instant of
execution. This poses a challenge to the initial construction of these *Instruments*.

*Rateslib* handles this by allowing a 2-tuple as an input to ``fx_fixings``. The first entry is
assigned to the first period and the latter entry is the FX fixings *Series*.

Consider the example below.

.. ipython:: python

   df = DataFrame(
       data=[1.19, 1.21, 1.24],
       index=Index(["17-01-2023", "17-04-2023", "17-07-2023"], name="reference_date"),
       columns=["rate"]
   )
   print(df)
   df.to_csv("gbpusd.csv")  # Save the DataFrame and create a CSV file

.. ipython:: python
   :okwarning:

   xcs = XCS(
       effective=dt(2023, 1, 15),
       termination="9M",
       spec="gbpusd_xcs",
       fx_fixings=(1.20, defaults.fixings["gbpusd"]),
   )
   xcs.cashflows(curves=curve, fx=1.25)  # arguments here used as a placeholder to display values.

Note how the rate for initial exchange is 1.20 (and not 1.19)
and the MTM payments are 1.21 and 1.24, as expected.
