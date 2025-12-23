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

The :class:`~rateslib.data.loader.DefaultFixingsLoader` provides a way for a user to load
fixings directly into *rateslib*. Data can be populated in one of **two** ways. Either,

1) Manually, with Python objects, using the :meth:`~rateslib.data.loader._BaseFixingsLoader.add`
   method. This is often used for simple examples.

2) From a datafile stored as a CSV. This must have a particular template format.

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

Now we set the ``directory`` of the default ``loader`` to point to this folder
and load the fixings directly.

.. ipython:: python

   fixings.loader.directory = os.getcwd()
   fixings["SEK_IBOR_3M"]

.. ipython:: python
   :suppress:

   import os
   os.remove("SEK_IBOR_3M.csv")

Note this *__getitem__* mechanism loads a state id, the timeseries itself and the range of the
index. When an attempt is made for an unavailable dataset a *ValueError* is raised.

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
      values. The below fixing has a *reference value date* of 12-Sep, with a 3-month lag and
      therefore is linearly interpolated between the Jun and July values and should be
      approximately 405.

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

      The relevant date for an *FX Fixing* is its *delivery date*.

      .. important::

         *FX Fixings* depend upon a named *series*, which defines the time the fixings are
         calculated, the calculation agent, and the methodology. The **required** fixings are
         the USD majors. Crosses are derived from those USD majors in order to avoid
         triangulation arbitrage.

      As an example, suppose the following values are published by Reuters at 4PM GMT
      for the spot (T+2) GBPUSD major in 2025.

      ===========  ==========  =======
      Publication  Spot        Value
      ===========  ==========  =======
      11th Sep     15th Sep    1.3574
      12th Sep     16th Sep    1.3556
      15th Sep     17th Sep    1.3599
      ===========  ==========  =======

      This data must be keyed by the format: *"{series_name}_{pair}"*  and is thus
      entered into *rateslib* as the following dates:

      .. ipython:: python

         values = Series(
             index=[dt(2025, 9, 15), dt(2025, 9, 16), dt(2025, 9, 17)],
             data=[1.3574, 1.3556, 1.3599]
         )
         fixings.add("REUTERS/4PMGMT/T+2_GBPUSD", values)

      This data is sufficient to populate some :class:`~rateslib.data.fixings.FXFixing`
      values.

      .. ipython:: python

         fx_fixing = FXFixing(
             date=dt(2025, 9, 16),
             identifier="REUTERS/4PMGMT/T+2",
             pair="gbpusd",
         )
         fx_fixing.value

      **Crosses**

      FX crosses require two USD majors to be populated to **the same series**. For example
      to determine a *GBPEUR* fixing we require the *EURUSD* rates.

      .. ipython:: python

         values = Series(
             index=[dt(2025, 9, 15), dt(2025, 9, 16), dt(2025, 9, 17)],
             data=[1.1112, 1.1234, 1.1199]
         )
         fixings.add("REUTERS/4PMGMT/T+2_EURUSD", values)

      .. ipython:: python

         fx_fixing = FXFixing(
             date=dt(2025, 9, 16),
             identifier="REUTERS/4PMGMT/T+2",
             pair="gbpeur",
         )
         fx_fixing.value  # <- cross of 1.3556 / 1.1234.

      The series *"REUTERS/4PMGMT/T+2_GBPEUR"*, if loaded to the fixings object, would **not**
      be used.

      **Misaligned dates**

      When majors are quoted with misaligned delivery dates, e.g. USDCAD is typically quoted T+1
      and GBPUSD is quoted T+2, the data must be entered according to the calculation agents
      adjustment criteria, e.g. adjusting by the FX Swap market. This highlights the importance of
      labeling series names with their settlement so that appropriate derivations can be made.

      .. ipython:: python

         fixings.add("REUTERS/4PMGMT/T+1_USDCAD", Series(index=[dt(2025, 9, 16)], data=[1.0450]))
         fixings.add("REUTERS/4PMGMT/T+2_USDCAD", Series(index=[dt(2025, 9, 16)], data=[1.0455]))
         fx_fixing = FXFixing(
             date=dt(2025, 9, 16),
             identifier="REUTERS/4PMGMT/T+2",
             pair="gbpcad",
         )
         fx_fixing.value  # <- cross of 1.3556 * 1.0455

      .. ipython:: python
         :suppress:

         fixings.pop("REUTERS/4PMGMT/T+2_EURUSD")
         fixings.pop("REUTERS/4PMGMT/T+1_USDCAD")
         fixings.pop("REUTERS/4PMGMT/T+2_USDCAD")
         fixings.pop("REUTERS/4PMGMT/T+2_GBPUSD")


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
