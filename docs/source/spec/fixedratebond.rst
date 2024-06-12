.. ipython:: python
   :suppress:

   from rateslib import *

**************
FixedRateBond
**************

USD
****

.. _spec-us-gb:

Government Bonds
------------------

Aliases: **"ust"**

Uses Street convention. Similar to *"uk_gb"* except long stub periods have linear
proportioning only in the segregated short stub part.

.. ipython:: python

   defaults.spec["us_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_gb", fixed_rate=2.5).kwargs

US Treasury convention. Reprices examples in federal documents: Section 31-B-ii).

.. ipython:: python

   defaults.spec["us_gb_tsy"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_gb_tsy", fixed_rate=2.5).kwargs

EUR
********

.. _spec-de-gb:

Government Bonds
-----------------

**Germany**

Uses ICMA conventions. Similar to *"uk_gb"*, except in the last period simple interest rate and
money-market yield is used.

.. ipython:: python

   defaults.spec["de_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="de_gb", fixed_rate=2.5).kwargs

.. _spec-fr-gb:

**France**

Uses ICMA conventions. Similar to *"uk_gb"*.

.. ipython:: python

   defaults.spec["fr_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="fr_gb", fixed_rate=2.5).kwargs

GBP
********

.. _spec-uk-gb:

Government Bonds
-----------------

Aliases: **"ukt"** and **"gilt"**

Calculations performed with the DMO method. Accrued is on ActAct linearly proportioned basis.
Yield is compounded in all periods including any front and back stubs.


.. ipython:: python

   defaults.spec["uk_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="uk_gb", fixed_rate=2.5).kwargs


SEK
*****

.. _spec-se-gb:

Government Bonds
-----------------

Aliases: **"sgb"**

Calculation performed with Swedish DMO method, using 30e360 for accrued calculations and for back stubs.

.. ipython:: python

   defaults.spec["se_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="se_gb", fixed_rate=2.5).kwargs


CAD
****

.. _spec-ca-gb:

Government Bonds
------------------

Aliases **"cadgb"**

Canadian government bond convention. Accrued is calculated using an ACT365F
convention. Yield calculations are still derived with linearly proportioned compounded
coupons.

.. ipython:: python

   defaults.spec["ca_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="ca_gb", fixed_rate=2.5).kwargs

