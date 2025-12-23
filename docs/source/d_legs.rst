.. _legs-doc:

***********
Legs
***********

Conventional Legs
*********************

The ``rateslib.legs`` module creates *Legs* which
typically contain a list of :ref:`Periods<periods-doc>`. The pricing, and
risk, calculations of *Legs* resolves to a linear sum of those same calculations
looped over all of the individual *Periods*.
Like *Periods*, it is probably quite
rare that *Legs* will be instantiated directly, rather they will form the
components of :ref:`Instruments<instruments-toc-doc>`, but none-the-less, this page
describes their construction.

.. autosummary::
   rateslib.legs.FixedLeg
   rateslib.legs.FloatLeg
   rateslib.legs.ZeroFixedLeg
   rateslib.legs.ZeroFloatLeg
   rateslib.legs.ZeroIndexLeg

Credit Legs
*************

.. autosummary::
   rateslib.legs.CreditPremiumLeg
   rateslib.legs.CreditProtectionLeg

Custom Legs and Objects
*****************************************

.. autosummary::
   rateslib.legs.CustomLeg
   rateslib.legs.Amortization
   rateslib.legs._BaseLeg

Protocols
***************

.. autosummary::
   rateslib.legs.protocols._WithNPV
   rateslib.legs.protocols._WithCashflows
   rateslib.legs.protocols._WithAnalyticDelta
   rateslib.legs.protocols._WithAnalyticRateFixings
   rateslib.legs.protocols._WithExDiv

