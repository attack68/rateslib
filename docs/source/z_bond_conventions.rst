.. _cook-bond_convs:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

How to customise FixedRateBond conventions
******************************************************

There are hundreds of bond conventions in use across different sectors and geographies.
*Rateslib* tries to provide a framework general enough to catch the most used conventions
by default, but flexible enough to provide a user with the ability to customise it to their
own requirements.

Setting up a :class:`~rateslib.instruments.FixedRateBond` in *rateslib* can be
done in different ways, with each method becoming more granular depending upon the necessary
level of customisation that is required to properly match the intended convention.

*Bonds* configured in *rateslib* have **two** types of convention parameters;

- **Regular scheduling parameters**, which are similar to *swaps* and contain
  arguments like ``calendar``, ``convention``, ``modifier``, ``ex_div``, relevant to cashflow
  and date determination.
- **Calculation modes** which are unique to *bonds* and define certain types of calculation
  per bond type. For example, how accrued interest is calculated for settlement and during
  YTM calculations, and how YTM calculations are performed.

Configured defaults
--------------------

There are a number of configured default specifications (``spec``) that have been setup.
The names of these can be found in the **securities** section of
:ref:`User Guide > Defaults <defaults-arg-input>`.

These are useful because they might be compared against to determine how a specific bond
convention might be setup. For example the following parameters are well
recognised values for a **US Treasury Bond**:

- ``convention``: *"ActActICMA"*,
- ``calendar``: *"nyc"*,
- ``modifier``: *"none"* (coupon dates are not modified according to holiday calendars)
- ``payment_lag``: 0 (payment will take place on or immediately after the coupon date).

These are visible directly in the *default dict* for a US government bond.

.. ipython:: python

   from rateslib import defaults
   defaults.spec["us_gb"]

One observes that the ``calc_mode`` here is also consistent for a *"us_gb"*.

Regular scheduling parameters
-------------------------------

Sometimes multiple scheduling parameters can result in the same cashflow periods.
For example, the below bond does not define any holidays in its calendar so no modification are
made to coupon dates. But that also means payment dates are not adjusted to real
business days either.

.. ipython:: python

   bond = FixedRateBond(dt(2000, 2, 17), "2y", fixed_rate=4.0, frequency="S", calendar="all", convention="actacticma")
   bond.cashflows()

A better configuration (which is reflected in *rateslib* defaults) is to directly specify
no *modifier* with an appropriate holiday calendar to adjust payments.

.. ipython:: python

   bond = FixedRateBond(dt(2000, 2, 17), "2y", fixed_rate=4.0, frequency="S", calendar="nyc", modifier="none", convention="actacticma")
   bond.cashflows()


Calculation modes
-------------------

The ``calc_mode`` argument allows a string input for a
:class:`~rateslib.instruments.bonds.conventions.BondCalcMode` that is predefined, *or*
a user can defined their own.

For the above US Treasury Bond the *calculation mode* is preconfigured and has the
following representation:

.. ipython:: python

   from rateslib.instruments.bonds.conventions import US_GB
   US_GB.kwargs

This differs from another convention, such as for a German Bund, which has the following
representation:

.. ipython:: python

   from rateslib.instruments.bonds.conventions import DE_GB
   DE_GB.kwargs

A :class:`~rateslib.instruments.bonds.conventions.BondCalcMode` can be directly constructed
and passed as the ``calc_mode`` in the *FixedRateBond* initialisation.
The relevant properties of the construction are explained on the documentation page for that
object. Some