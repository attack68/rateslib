.. _fixings-doc:

.. ipython:: python
   :suppress:

   from rateslib import fixings, dt
   from rateslib.data.fixings import IndexFixing, FXFixing, IBORFixing, IBORStubFixing, FloatRateIndex, FloatRateSeries, RFRFixing
   from pandas import DataFrame, Index, Series
   import os

Fixings
===========

Published financial fixings form a key part of the valuation of *Instruments* and for those used
in *Curve* or *Surface* calibration. *Rateslib* cannot be distributed with financial data,
therefore data management must be handled externally by the user.

But, some easy-to-use classes exist to provide the bridge between external data and the
objects constructed in *rateslib*. The **global** ``fixings`` object coordinates
this interaction and is an instance of the :class:`~rateslib.data.loader.Fixings` class.

The key attribute of this class is its ``loader`` which is, by default, an instance of the
:class:`~rateslib.data.loader.DefaultFixingsLoader`, but a user can replace this with their
own implementation to interact with their own data sources.

DefaultFixingsLoader
----------------------

The :class:`~rateslib.data.loader.DefaultFixingsLoader` can be populated with fixings data in
**two** ways. Either,

1) From a datafile stored as a CSV. This must have a particular template format.

2) Manually, with Python objects, using the :meth:`~rateslib.data.loader._BaseFixingsLoader.add`
   method.

Let's demonstrate the CSV method. Create a CSV in the required date-string format in the
current working directory.

.. ipython:: python

   df = DataFrame(
       data=[1.21, 2.75],
       index=Index(["02-01-2023", "03-01-2023"], name="reference_date"),
       columns=["rate"]
   )
   print(df)
   df.to_csv("SEK_IBOR_3M.csv")  #  <--- Save a CSV file to disk

Now we set the ``directory`` of the ``loader`` to point to this folder
and load the fixings directly.

.. ipython:: python

   fixings.loader.directory = os.getcwd()
   fixings["SEK_IBOR_3M"]

.. ipython:: python
   :suppress:

   import os
   os.remove("SEK_IBOR_3M.csv")

Notice this get item mechanism loads a state id, the timeseries itself and the range of the index.
When an attempt is made for an unavailable dataset a *ValueError* is raised.

.. ipython:: python

   try:
       fixings["unavailable_data"]
   except ValueError as e:
       print(e)

It is also possible to add a dataseries created in Python directly to the ``fixings`` object.

.. ipython:: python

   ts = Series(index=[dt(2000, 1, 1), dt(2000, 2, 1)], data=[666., 667.])
   fixings.add("my_series", ts)
   fixings["MY_SERIES"]

Any data can be directly removed

.. ipython:: python

   fixings.pop("My_Series")

Lowercase and uppercase are ignored.

Relevant dates
---------------

To date, *rateslib* makes use of 3 classifications of fixing; **index**, **fx** and **rate**.

.. tabs::

   .. tab:: Index

      The use case for an :class:`~rateslib.data.fixings.IndexFixing` is for inflation
      related *Instruments*.

      The relevant date for an :class:`~rateslib.data.fixings.IndexFixing` is its *reference value date*.
      Determining the ``value`` of an :class:`~rateslib.data.fixings.IndexFixing` depends upon other
      information such as the ``index_lag``, and ``index_method``.

      .. important::

         Inflation data, and other monthly data, must be indexed by the **start of the month** relating
         to the data publication.

      As an example the following data in 2025 for the UK RPI series was released:

      ===========  ======  =======
      Publication  Month   Value
      ===========  ======  =======
      18th June    May     402.9
      16th July    June    404.5
      20th Aug     July    406.2
      ===========  ======  =======

      This must be entered into *rateslib* as the following dates:

      .. ipython:: python

         uk_rpi = Series(
             index=[dt(2025, 5, 1), dt(2025, 6, 1), dt(2025, 7, 1)],
             data=[402.9, 404.5, 406.2]
         )
         fixings.add("UK_RPI", uk_rpi)

      This data is then sufficient to populate some :class:`~rateslib.data.fixings.IndexFixing`
      values.

      .. ipython:: python

         index_fixing = IndexFixing(
             index_lag=3,
             index_method="daily",
             date=dt(2025, 9, 12),
             identifier="UK_RPI"
         )
         index_fixing.value

   .. tab:: FX

      The use case for an :class:`~rateslib.data.fixings.FXFixing` is multi-currency *Instruments*.

      The relevant date for an :class:`~rateslib.data.fixings.FXFixing` is its *delivery date*.

      .. tip::

         Different *'cuts'* can be populated simply by creating separate timeseries and
         identifying them by a separate name, e.g. 'GBPUSD_1600hrs' and 'GBPUSD_1000hrs'

      As an example, suppose the following 4PM GMT GBPUSD Spot FX rates were recorded in 2025.

      ===========  ==========  =======
      Publication  Spot        Value
      ===========  ==========  =======
      11th Sep     15th Sep    1.3574
      12th Sep     16th Sep    1.3556
      15th Sep     17th Sep    1.3599
      ===========  ==========  =======

      This data is entered into *rateslib* as the following dates:

      .. ipython:: python

         gbpusd_1600_gmt = Series(
             index=[dt(2025, 9, 15), dt(2025, 9, 16), dt(2025, 9, 17)],
             data=[1.3574, 1.3556, 1.3599]
         )
         fixings.add("GBPUSD_1600_GMT", gbpusd_1600_gmt)

      This data is sufficient to populate some :class:`~rateslib.data.fixings.FXFixing`
      values.

      .. ipython:: python

         fx_fixing = FXFixing(
             date=dt(2025, 9, 16),
             identifier="GBPUSD_1600_GMT"
         )
         fx_fixing.value

   .. tab:: Rates

      .. important::

         The relevant date for an :class:`~rateslib.data.fixings.IBORFixing` or
         :class:`~rateslib.data.fixings.IBORStubFixing` is its **publication date**, whilst the
         relevant date for an :class:`~rateslib.data.fixings.RFRFixing` is its **reference value**
         date. These difference generally reflect market conventions for handling this data.

      Rates fixings are handled slightly differently to other fixings types since there is the
      notion of a :class:`~rateslib.data.fixings.FloatRateSeries` which is a particular set of
      conventions for multiple, different tenor :class:`~rateslib.data.fixings.FloatRateIndex`.

      .. important::

         When rates fixings are populated, their identifier **must contain a suffix** which matches
         the frequency of the index. Internally *rateslib* uses these identifiers to match the
         right fixing frequency to the right dataset.

      **IBOR**

      Suppose the following data is observed for EURIBOR in 2025

      ============  ==============  ============  ==========
      Publication   Value Date      EURIBOR 3M    EURIBOR 6M
      ============  ==============  ============  ==========
      11th Sep      15th Sep        2.014         2.119
      12th Sep      16th Sep        2.000         2.108
      15th Sep      17th Sep        2.033         2.101
      ============  ==============  ============  ==========

      This is populated to *rateslib* by publication date as follows,

      .. ipython:: python

         euribor_3m = Series(
             index=[dt(2025, 9, 11), dt(2025, 9, 12), dt(2025, 9, 15)],
             data=[2.014, 2.000, 2.033]
         )
         euribor_6m = Series(
             index=[dt(2025, 9, 11), dt(2025, 9, 12), dt(2025, 9, 15)],
             data=[2.119, 2.108, 2.101]
         )
         fixings.add("EURIBOR_3M", euribor_3m)
         fixings.add("EURIBOR_6M", euribor_6m)

      These can populate the the following fixing values.

      .. ipython:: python

         ibor_fixing=IBORFixing(
             accrual_start=dt(2025, 9, 16),
             identifier="EURIBOR_3M",
             rate_index=FloatRateIndex(
                 frequency="Q",
                 series="eur_ibor"
             )
         )
         ibor_fixing.value

         ibor_stub_fixing=IBORStubFixing(
             accrual_start=dt(2025, 9, 16),
             accrual_end=dt(2026, 1, 22),
             identifier="EURIBOR",
             rate_series="eur_ibor",
         )
         ibor_stub_fixing.value

      Note that these individual objects also store attributes that might be useful to inspect
      (see the individual class documentation). For example

      .. ipython:: python

         ibor_stub_fixing.fixing1.value
         ibor_stub_fixing.fixing2.value
         ibor_stub_fixing.weights

      **RFR**

      An :class:`~rateslib.data.fixings.RFRFixing` is a determined rate fixing for an entire
      *Period*, not just one single RFR publication. This means that it must account for the
      ``fixing_method``, ``spread_compound_method`` and ``float_spread``. If there are not
      sufficient publications to determine a full *Period*  fixing it will not be assigned a value.

      Suppose the following data is observed for ESTR in 2025

      ============  ==============  ============
      Publication   Value Date      ESTR 1B
      ============  ==============  ============
      15th Sep      12th Sep        1.91
      16th Sep      15th Sep        1.92
      17th Sep      16th Sep        1.93
      ============  ==============  ============

      This is populated to *rateslib* by reference value date as follows,

      .. ipython:: python

         estr_1b = Series(
             index=[dt(2025, 9, 12), dt(2025, 9, 15), dt(2025, 9, 16)],
             data=[1.91, 1.92, 1.93]
         )
         fixings.add("ESTR_1B", estr_1b)

      The following *Period* is fully specified and will determine a value.

      .. ipython:: python

         rfr_fixing = RFRFixing(
             accrual_start=dt(2025, 9, 12),
             accrual_end=dt(2025, 9, 19),
             identifier="ESTR_1B",
             spread_compound_method="NoneSimple",
             fixing_method="RFRLockout",
             method_param=2,
             float_spread=100.0,
             rate_index=FloatRateIndex(frequency="1B", series="eur_rfr")
         )
         rfr_fixing.value

      Again some attributes may be useful to inspect.

      .. ipython:: python

         rfr_fixing.populated

      A *Period* which is not fully specified will not return a value, but may still contain
      information regarding some of the individual fixings related to the period.

      .. ipython:: python

         rfr_fixing = RFRFixing(
             accrual_start=dt(2025, 9, 12),
             accrual_end=dt(2025, 9, 19),
             identifier="ESTR_1B",
             spread_compound_method="NoneSimple",
             fixing_method="RFRPaymentDelay",
             method_param=0,
             float_spread=100.0,
             rate_index=FloatRateIndex(frequency="1B", series="eur_rfr")
         )
         rfr_fixing.value
         rfr_fixing.populated
