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

Uses Street convention. Similar to *"uk_gb"* except long stub periods have linear
proportioning only in the segregated short stub part.

.. ipython:: python

   defaults.spec["us_gb"]
   from rateslib.instruments.components.bonds.conventions import US_GB
   US_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_gb", fixed_rate=2.5).kwargs

US Treasury convention. Reprices examples in federal documents: Section 31-B-ii).

.. ipython:: python

   defaults.spec["us_gb_tsy"]
   from rateslib.instruments.components.bonds.conventions import US_GB_TSY
   US_GB_TSY.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_gb_tsy", fixed_rate=2.5).kwargs

.. _spec-us-corp:

Corporate Bonds
----------------

.. ipython:: python

   defaults.spec["us_corp"]
   from rateslib.instruments.components.bonds.conventions import US_CORP
   US_CORP.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_corp", fixed_rate=2.5).kwargs

.. _spec-us-muni:

Municipal Bonds
-----------------

.. ipython:: python

   defaults.spec["us_muni"]
   from rateslib.instruments.components.bonds.conventions import US_MUNI
   US_MUNI.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="us_muni", fixed_rate=2.5).kwargs

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
   from rateslib.instruments.components.bonds.conventions import DE_GB
   DE_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="de_gb", fixed_rate=2.5).kwargs

.. _spec-fr-gb:

**France**

Uses ICMA conventions. Similar to *"uk_gb"*.

.. ipython:: python

   defaults.spec["fr_gb"]
   from rateslib.instruments.components.bonds.conventions import FR_GB
   FR_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="fr_gb", fixed_rate=2.5).kwargs

.. _spec-it-gb:

**Italy**

Coupons are semi-annual but yield convention is annual yield. In last coupon period simple yield is applied.

.. ipython:: python

   defaults.spec["it_gb"]
   from rateslib.instruments.components.bonds.conventions import IT_GB
   IT_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="it_gb", fixed_rate=2.5).kwargs

.. _spec-nl-gb:

**Netherlands**

Street convention is used, except when the bond is in the final coupon period simple interest yield is used.

.. ipython:: python

   defaults.spec["nl_gb"]
   from rateslib.instruments.components.bonds.conventions import NL_GB
   NL_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="nl_gb", fixed_rate=2.5).kwargs

CHF
********

.. _spec-ch-gb:

Government Bonds
-----------------

Calculations performed with ICMA convention.


.. ipython:: python

   defaults.spec["ch_gb"]
   from rateslib.instruments.components.bonds.conventions import CH_GB
   CH_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="ch_gb", fixed_rate=2.5).kwargs

GBP
********

.. _spec-uk-gb:

Government Bonds
-----------------

Calculations performed with the DMO method. Accrued is on ActAct linearly proportioned basis.
Yield is compounded in all periods including any front and back stubs.

.. ipython:: python

   defaults.spec["uk_gb"]
   from rateslib.instruments.components.bonds.conventions import UK_GB
   UK_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="uk_gb", fixed_rate=2.5).kwargs


SEK
*****

.. _spec-se-gb:

Government Bonds
-----------------

Calculation performed with Swedish DMO method, using 30e360 for accrued calculations and for back stubs.

.. ipython:: python

   defaults.spec["se_gb"]
   from rateslib.instruments.components.bonds.conventions import SE_GB
   SE_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="se_gb", fixed_rate=2.5).kwargs

NOK
****

.. _spec-no-gb:

Government Bonds
----------------

Using annualised yield calculation under ICMA compounding convention. Stub periods use ACT365 day fraction.
Accrual is calculated with ACT365F.

.. ipython:: python

   defaults.spec["no_gb"]
   from rateslib.instruments.components.bonds.conventions import NO_GB
   NO_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="no_gb", fixed_rate=2.5).kwargs

CAD
****

.. _spec-ca-gb:

Government Bonds
------------------

Canadian government bond convention. Accrued is calculated using an ACT365F
convention. Yield calculations are still derived with linearly proportioned compounded
coupons. **Note** this is not the appropriate convention for monthly-pay securities.

.. ipython:: python

   defaults.spec["ca_gb"]
   from rateslib.instruments.components.bonds.conventions import CA_GB
   CA_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="ca_gb", fixed_rate=2.5).kwargs

NZD
****

.. _spec-nz-gb:

Government Bonds
-------------------

Kiwi government bond convention.
Yield calculations use normal conventions except in the case of one coupon
payment remaining, when simple ACT365F is used.
See `Bond Memorandum <https://debtmanagement.treasury.govt.nz/sites/default/files/2024-11/nz-govt-nominal-bonds-information-memorandum-5nov24.pdf>`_

.. ipython:: python

   defaults.spec["nz_gb"]
   from rateslib.instruments.components.bonds.conventions import NZ_GB
   NZ_GB.kwargs
   FixedRateBond(dt(2000, 1, 1), "10y", spec="nz_gb", fixed_rate=2.5).kwargs
