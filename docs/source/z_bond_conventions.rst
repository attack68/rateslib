.. _cook-bond_convs:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Understanding and Customising FixedRateBond Conventions
********************************************************

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

One observes that the ``calc_mode`` here is also consistent for a *"us_gb"*, which we review later.

Regular scheduling parameters
-------------------------------

Sometimes multiple scheduling parameters can result in the same cashflow periods.
For example, the below bond does not define any holidays in its calendar so no modifications are
made to coupon dates. But that also means payment dates are not adjusted to real
business days either.

.. ipython:: python

   bond = FixedRateBond(dt(2000, 2, 17), "2y", fixed_rate=4.0, frequency="S", calendar="all", convention="actacticma")
   bond.cashflows()

A better configuration (which is reflected in *rateslib* defaults) is to directly specify
a *modifier* of *"none"* but with an appropriate holiday calendar to adjust physical payment dates.

.. ipython:: python

   bond = FixedRateBond(dt(2000, 2, 17), "2y", fixed_rate=4.0, frequency="S", calendar="nyc", modifier="none", convention="actacticma")
   bond.cashflows()


Calculation modes
-------------------

The ``calc_mode`` argument is the element that gives more direct control of calculations.
It allows a string input for a
:class:`~rateslib.instruments.bonds.conventions.BondCalcMode` that is predefined, *or*
a user can define their own.

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

A :class:`~rateslib.instruments.BondCalcMode` can be directly constructed
and passed as the ``calc_mode`` in the *FixedRateBond* initialisation.
The relevant properties of the construction are explained on the documentation page for that
object. It contains all of the necessary formulae to achieve the desired results. Importantly
all the functions must be correctly specified, or implemented, such that each element of the YTM
formula (visible in the docs for :class:`~rateslib.instruments.BondCalcMode`) are calculable.

Example implementation
------------------------

*Rateslib* has **not** implemented Thai Government Bonds by default, but let's suppose we want to
construct one. The calculation for these types of bonds were found in a document on the Thai
Bond Market Association website (:download:`pdf copy <api/_static/thai_standard_formula.pdf>`)

An example (A-3) is given which provides a couple of actionable tests.
The ``convention`` for Thai GBs uses
Act365F and the accrued interest matches this convention with Act365F, so a ``linear_days`` accrual
function will return an accrual fraction that determines the correct accrued interest. Noting,

.. math::

   \underbrace{\frac{r_u}{s_u}}_{\text{accrual fraction}} \underbrace{\frac{s_u}{365} C}_{\text{cashflow}} = \underbrace{\frac{r_u}{365} C}_{\text{accrued interest formula}}

Since ``linear_days`` is the default, the correct amount of accrued interest should be returned
by default when constructing a bond with an Act365F *convention*. The official example
gives an accrued interest calculation of 4.86986301. *Rateslib* gives the following:

.. ipython:: python

   bond = FixedRateBond(
       effective=dt(1991, 1, 15),
       termination=dt(1996, 4, 30),
       stub="shortback",
       fixed_rate=11.25,
       frequency="S",
       roll=15,
       convention="act365f",
       modifier="none",
       currency="thb",
       calendar="bus",
   )
   bond.accrued(settlement=dt(1994, 12, 20))

The calculations for YTM are not as straightforward, however. The official example gives the
clean price for a YTM of 8.75% to be 103.1099263, however, *rateslib* default
calculation mode returns:

.. ipython:: python

   bond.price(ytm=8.75, settlement=dt(1994, 12, 20))

From the specific Thai YTM formula this is due to a number of things.
Firstly, the discount functions,
*v1* and *v3* are handling the days in the stubs differently to Thai conventions.
To match, these must be implemented directly.

.. ipython:: python

   def _v1_thb_gb(
       obj,         # the bond object
       ytm,         # y as defined
       f,           # f as defined
       settlement,  # datetime
       acc_idx,     # the index of the period in which settlement occurs
       v2,          # the numeric value of v2 already calculated
       accrual,     # the ytm_accrual function to return accrual fractions
       period_idx,  # the index of the current period
   ):
       """The exponent to the regular discount factor is derived from ACT365F"""
       r_u = (obj.leg1.schedule.uschedule[acc_idx + 1] - settlement).days
       return v2 ** (r_u * f / 365)

.. ipython:: python

   def _v3_thb_gb(obj, ytm, f, settlement, acc_idx, v2, accrual, period_idx):
       """The exponent to the regular discount function is derived from ACT365F"""
       r_u = (obj.leg1.schedule.uschedule[-1] - obj.leg1.schedule.uschedule[-2]).days
       return v2 ** (r_u * f / 365)

Lastly, the Thai YTM formula assumes a standardised coupon payment for the regular flows, whereas
the actual convention of Act365F does not generate the same, standardised coupon payments
each period. This is also amended from default by setting the ``c1_type`` and
``ci_type`` to be ``full_coupon``. The back stub remains as ``cashflow``.

With these modifications to the ``calc_mode`` the bond returns exactly that which aligns with
the official source.

.. ipython:: python

   from rateslib.instruments import BondCalcMode
   thb_gb = BondCalcMode(
       settle_accrual="linear_days",
       ytm_accrual="linear_days",
       v1=_v1_thb_gb,
       v2="regular",
       v3=_v3_thb_gb,
       c1="full_coupon",
       ci="full_coupon",
       cn="cashflow",
   )
   bond = FixedRateBond(
       effective=dt(1991, 1, 15),
       termination=dt(1996, 4, 30),
       stub="shortback",
       fixed_rate=11.25,
       frequency="S",
       roll=15,
       convention="act365f",
       modifier="none",
       currency="thb",
       calendar="bus",
       calc_mode=thb_gb
   )
   bond.accrued(settlement=dt(1994, 12, 20))

.. ipython:: python

   bond.price(ytm=8.75, settlement=dt(1994, 12, 20))

These conventions work specifically for this bond because it was identified that it had a
back stub, but for the more general case it would be better to implement and pass
custom cashflow functions with a name similar to *'full_coupon_except_cashflow_stub'*.
