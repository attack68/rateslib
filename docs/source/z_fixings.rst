.. _cook-fixings-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Index
   import os

Working with Fixings
**********************

..  The default *Fixings* object
    ----------------------------

    Financial fixing data is important for financial *Instrument* specification and pricing.
    *Rateslib* uses a global :class:`~rateslib.default.Defaults` object and it stores its fixing
    data, that is internally referenced, under the attribute ``fixings``. *Rateslib* provides a default
    object, :class:`~rateslib.fixings.Fixings`, as the bridge between loading stored data and
    populating *Instrument* pricing data when required.

    Currently there are two methods for populating data using the :class:`~rateslib.fixings.Fixings` class.

    1) Creating CSV files and loading from file. The example below creates a CSV dynamically using
       Python, storing it in the relevant directory and then loads the fixing series directly.

    .. ipython:: python

       # Set the fixings default directoyr
       defaults.fixings.directory = os.getcwd()

       # Create a CSV file in the correct and unique format.
       df = DataFrame({
           "reference_date": ["01-01-2000", "01-02-2000", "01-03-2000"],
           "rate": [100.0, 101.1, 102.2]
       }).set_index("reference_date")
       df.to_csv("CPI_EXAMPLE.csv")

       # Load that file to a fixings Series object.
       defaults.fixings["CPI_EXAMPLE"]

    2) Constructing Series objects in Python and directly and adding them with the
       :meth:`~rateslib.fixings.Fixings.add_series` method. Here we will also remove the above example
       to demonstrate simple overwrites.

    .. ipython::

       # Remove the existing data (the CSV file is not deleted), and modify the data in the Series.
       popped = defaults.fixings.remove_series("CPI_EXAMPLE").astype(object)
       popped[dt(2000, 2, 1)] = Dual(101.1, ["x"], [])

       # Add the modified Series with a new ID.
       defaults.fixings.add_series("CPI_EXAMPLE", popped)
       defaults.fixings["CPI_EXAMPLE"]

    The fixings *Series* are lazily loaded, and once loaded will be stored in the dictionary
    they will not be reloaded unless removed using :meth:`~rateslib.fixings.Fixings.remove_series`
    and re-called.



    Building a custom Fixings loader
    --------------------------------

    The class :class:`~rateslib.fixings._BaseFixingsLoader` is the abstract base class required to
    build and insert a custom fixings loader. For example, suppose you did not want to use CSV files,
    but instead wanted a mechanism to call some SQL and fetch a fixing series from a database.

    This is possible by overloading *rateslib's* :class:`~rateslib.fixings.Fixings` object.
    The following object is a simplified construction of this idea.


    .. note::

       To understand the difference between RFR and IBOR date indexing in *rateslib* see
       :ref:`IBOR or RFR <c-curves-ibor-rfr>`.

    The *rateslib* `defaults` object lazy loads fixings from CSV files.
    These files are stored in the rateslib package files under the directory *'data'*.
    Due to static packaging and licencing issues rateslib cannot be distributed
    with accurate and upto date RFR or IBOR fixings.

    As an example, a small selection of SOFR fixings which are used in some examples
    within, *rateslib* documentation are shown below along with the source directory.


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
       from rateslib.scheduling import get_calendar
       nyc = get_calendar("nyc")
       fixings = Series(
           data=2.5,
           index=nyc.bus_date_range(start=dt(2022, 1, 3), end=dt(2022, 4, 14)),
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

    .. ipython:: python
       :suppress:

       import os
       os.remove("gbpusd.csv")
       os.remove("sek_ibor_3m.csv")
       os.remove("CPI_EXAMPLE.csv")


    .. ipython:: python

       class SQLLoader(_BaseFixingsLoader):

           def __init__(self) -> None:
               self.cache: dict[str, Series[float]] = {}

           def __getitem__(self, name: str):
               if name in self.cache:
                   return self.cache[name]
               else:
                   # simulate fetching some data from an SQL database.
                   from random import random
                   self.cache[name] = Series(
                       index=[dt(2000, 1, 1), dt(2001, 1 ,1)],
                       data=[random(), random()],
                   )
                   return self.__getitem__(name)

    Now overload the *rateslib* object and get fixings as usual:

    .. ipython:: python

       defaults.fixings = SQLLoader()
       defaults.fixings["RANDOM_INDEX"]