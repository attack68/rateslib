.. _legs-doc:

.. ipython:: python
   :suppress:

   from rateslib.periods import *
   from rateslib.legs import *
   from rateslib.scheduling import Schedule
   from datetime import datetime as dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
   )

****
Legs
****

The ``rateslib.legs`` module creates *Legs* which
typically contain a list of :ref:`Periods<periods-doc>`. The pricing, and
risk, calculations of *Legs* resolves to a linear sum of those same calculations
looped over all of the individual *Periods*.
Like *Periods*, it is probably quite
rare that *Legs* will be instantiated directly, rather they will form the
components of :ref:`Instruments<instruments-toc-doc>`, but none-the-less, this page
describes their construction.

The following *Legs* are provided, click on the links for a full description of each
*Leg* type:

.. inheritance-diagram::   rateslib.legs.BaseLeg rateslib.legs.BaseLegMtm rateslib.legs.FixedLeg rateslib.legs.FloatLeg rateslib.legs.IndexFixedLeg rateslib.legs.ZeroFloatLeg rateslib.legs.ZeroFixedLeg rateslib.legs.ZeroIndexLeg rateslib.legs.FixedLegMtm rateslib.legs.FloatLegMtm rateslib.legs.CreditProtectionLeg rateslib.legs.CreditPremiumLeg rateslib.legs.CustomLeg
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.legs.FixedLeg
   rateslib.legs.FloatLeg
   rateslib.legs.IndexFixedLeg
   rateslib.legs.ZeroFloatLeg
   rateslib.legs.ZeroFixedLeg
   rateslib.legs.ZeroIndexLeg
   rateslib.legs.CreditProtectionLeg
   rateslib.legs.CreditPremiumLeg
   rateslib.legs.CustomLeg

*Legs*, similar to *Periods*, are defined as having the following protocols:

TBD TODO


