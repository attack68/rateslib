.. _cook-reverse-xcs-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from datetime import datetime as dt

Configuring Cross-Currency Swaps - is it USDCAD or CADUSD?
*************************************************************

A Cross-Currency Swap (:class:`~rateslib.instruments.XCS`) is composed of two
*Legs*. Each leg is a different currency and in some cases it does not matter
which currency in the pair one assigns to which *Leg*.

However, for some kinds of *XCS* and for some features it does matter.

Mark-to-Market Swaps
---------------------

*MTM-XCSs* are the default interbank *XCS* instrument. These *Instruments* have a single *Leg*
whose notional is variable and is defined by future FX fixings. In *rateslib* **only Leg2**
can be defined as a MTM *Leg*. So whichever currency has the variable notional, and in general
it is the dominant currency, e.g. USD or EUR where USD is not present, must be assigned to *Leg2*.

Thus, irrespective of how one refers to a USDCAD or CADUSD XCS, if the variable notional leg is USD this
instrument should be constructed as a CADUSD instrument in *rateslib*.

Defining ``leg2_notional``
---------------------------

``leg2_notional`` cannot be specified directly for a :class:`~rateslib.instruments.XCS`,
instead it is implicitly calculated from ``notional`` (on *leg1*) and ``fx_fixings``.
For mid-market pricing, if ``fx_fixings`` is not initially given, these will be calculated
dynamically from an :class:`~rateslib.fx.FXForwards` object (as well as future FX fixings
for MTM legs).

If you were to trade a USDCAD MTM-XCS in US\$100mm with an initial FX fixing of USDCAD 1.35
implying a CA\$ 135mm notional, this should not be entered as specified here. Instead it should
be entered as a CADUSD MTM-XCS (with MTM Leg in USD) specified by a ``notional`` of 135mm and
an initial ``fx_fixings`` of 0.74074 (=1/1.35).

.. ipython:: python

   xcs = XCS(
       effective=dt(2000,1,1),
       termination="1y",
       frequency="q",
       notional=135e6,
       leg2_fx_fixings=0.7407407407407407,
       leg2_mtm=True,
       currency="cad",
       leg2_currency="usd",
    )

To see the cashflows of a *XCS* ``curves`` and ``fx`` are always required. Here we ignore
these to get a feel for the structure of this *Instrument*.

.. ipython:: python

   xcs.cashflows()

Calculating the ``rate``
---------------------------

The :meth:`XCS.rate <rateslib.instruments.XCS.rate>` method can calculate the implied rate for
any kind of *XCS*, whether it is parameterised as *Float/Float*, *Fixed/Float*, *Float/Fixed*, *Fixed/Fixed*
and *MTM* or *non-MTM*. By default it will calculate the spread on *Leg1* which is traditionally
where the basis-point spread is assigned (to the *non-MTM Leg*). For a CADUSD *XCS* this
would align with market standard quotations.
