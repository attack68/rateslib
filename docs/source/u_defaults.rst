.. _defaults-doc:

.. ipython:: python
   :suppress:

   from rateslib import *

Defaults
===========

Argument input management
-------------------------

The ``rateslib.default`` module is provided to give the user global control over a lot of
parameters that are set by default when the user provides no input to those arguments.

Since financial instrument specification usually contains a large number of parameters (for
example a cross-currency basis swap (:class:`~rateslib.instruments.XCS`) has around 50
possible arguments to initialise the swap with), argument management is a key part of the
:ref:`design philosophy<pillars-doc>` of *rateslib*.

The easiest way to construct conventional instruments is to use the ``spec`` argument.
A number of market conventions have already been pre-added to
*rateslib*, and if defining an *Instrument* that allows the ``spec`` (specification) argument
a host of arguments will be pre-populated. The table below outlines all of the existing
``spec`` arguments.

.. warning::

   When using the ``spec`` argument, arguments for *leg2* which might normally inherit
   from *leg1* might be defined specifically, and will no longer inherit. If overwriting
   an instrument that has been directly specified, ensure to overwrite both legs.

**Derivatives**

.. list-table::
   :header-rows: 1
   :class: scrollwide

   * - Currency
     - Calendar
     - IRS
     - XCS
     - FRA
     - SBS
     - STIRFuture
     - ZCS
     - ZCIS
     - IIRS
     - FXSwap
     - FXOption
   * - USD
     - :ref:`nyc <spec-usd-nyc>`, :ref:`fed <spec-usd-fed>`
     - :ref:`usd_irs <spec-usd-irs>`, :ref:`usd_irs_lt_2y <spec-usd-irs-lt-2y>`
     -
     - -
     - -
     -
     -
     -
     -
     -
     -
   * - EUR
     - :ref:`tgt <spec-eur-tgt>`
     - :ref:`eur_irs <spec-eur-irs>`, :ref:`eur_irs3 <spec-eur-irs3>`, :ref:`eur_irs6 <spec-eur-irs6>`, :ref:`eur_irs1 <spec-eur-irs1>`
     - :ref:`eurusd_xcs <spec-eurusd-xcs>`, :ref:`eurgbp_xcs <spec-eurgbp-xcs>`
     - :ref:`eur_fra3 <spec-eur-fra3>`, :ref:`eur_fra6 <spec-eur-fra6>`, :ref:`eur_fra1 <spec-eur-fra1>`
     - :ref:`eur_sbs36 <spec-eur-sbs36>`
     -
     -
     -
     -
     -
     -
   * - GBP
     - :ref:`ldn <spec-gbp-ldn>`
     - :ref:`gbp_irs <spec-gbp-irs>`
     - :ref:`gbpusd_xcs <spec-gbpusd-xcs>`, :ref:`gbpeur_xcs <spec-gbpeur-xcs>`
     - -
     - -
     -
     -
     -
     -
     -
     -
   * - CHF
     - :ref:`zur <spec-chf-zur>`
     - :ref:`chf_irs <spec-chf-irs>`
     -
     - -
     - -
     -
     -
     -
     -
     -
     -
   * - SEK
     - :ref:`stk <spec-sek-stk>`
     - :ref:`sek_irs <spec-sek-irs>`, :ref:`sek_irs3 <spec-sek-irs3>`
     -
     - :ref:`sek_fra3 <spec-sek-fra3>`
     -
     -
     -
     -
     -
     -
     -
   * - NOK
     - :ref:`osl <spec-nok-osl>`
     - :ref:`nok_irs <spec-nok-irs>`, :ref:`nok_irs3 <spec-nok-irs3>`, :ref:`nok_irs6 <spec-nok-irs6>`
     -
     - :ref:`nok_fra3 <spec-nok-fra3>`, :ref:`nok_fra6 <spec-nok-fra6>`
     - :ref:`nok_sbs36 <spec-nok-sbs36>`
     -
     -
     -
     -
     -
     -
   * - CAD
     - :ref:`tro <spec-cad-tro>`
     - :ref:`cad_irs <spec-cad-irs>`, :ref:`cad_irs_le_1y <spec-cad-irs-le-1y>`
     -
     - -
     - -
     -
     -
     -
     -
     -
     -
   * - JPY
     - :ref:`tyo <spec-jpy-tyo>`
     - :ref:`jpy_irs <spec-jpy-irs>`
     - :ref:`jpyusd_xcs <spec-jpyusd-xcs>`
     - -
     - -
     -
     -
     -
     -
     -
     -
   * - AUD
     - :ref:`syd <spec-aud-syd>`
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -

**Securities**

.. list-table::
   :header-rows: 1
   :class: scrollwide

   * - Currency
     - FixedRateBond
     - Bill
     - IndexFixedRateBond
     - FloatRateNote
   * - USD
     - :ref:`us_gb <spec-us-gb>`, :ref:`us_gb_tsy <spec-us-gb>`
     - :ref:`us_gbb <spec-usd-gbb>`
     -
     -
   * - EUR
     - :ref:`de_gb <spec-de-gb>`, :ref:`fr_gb <spec-fr-gb>`, :ref:`it_gb <spec-it-gb>`, :ref:`nl_gb <spec-nl-gb>`
     -
     -
     -
   * - GBP
     - :ref:`uk_gb <spec-uk-gb>`
     -
     -
     -
   * - CHF
     -
     -
     -
     -
   * - SEK
     - :ref:`se_gb <spec-se-gb>`
     -
     -
     -
   * - NOK
     - :ref:`no_gb <spec-no-gb>`
     -
     -
     -
   * - CAD
     - :ref:`ca_gb <spec-ca-gb>`
     -
     -
     -
   * - JPY
     -
     -
     -
     -
   * - AUD
     -
     -
     -
     -


.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    spec/calendars.rst
    spec/irs.rst
    spec/xcs.rst
    spec/sbs.rst
    spec/fra.rst
    spec/fixedratebond.rst
    spec/bill.rst


The NoInput argument
*********************

.. warning::

   When an argument is not provided this actually assumes a defined datatype in
   *rateslib* called :class:`~rateslib.default.NoInput`. **Never** use *None* as an entry to
   an argument, this will typically create downstream errors. It is better to omit the argument
   entry entirely and let *rateslib* control the *NoInput* value.

There are 3 types of :class:`~rateslib.default.NoInput` that work behind the scenes:

- **NoInput.blank**: this specifies the user has provided no input for this argument and if there
  is a default value that will be used instead. For example, not providing a ``convention`` will
  result in the value of ``defaults.convention`` being used.
- **NoInput.inherit**: this specifies that the user has provided no input for this argument and
  its value will be inherited from the equivalent attribute on ``leg1``. For example the
  value ``leg2_payment_lag`` has a value of *NoInput.inherit* meaning its value will be obtained
  from the value of ``payment_lag`` whether that is taken by default or set by a user.
- **NoInput.negate**: this is similar to *NoInput.inherit* except it negates the value. This is
  useful for ``notional`` and ``amortization`` when 2 legs commonly take opposite values.

In the below code snippet one can observe how these *NoInputs* are operating in the initialisation
of a swap to infer what a user might expect when just inputting a small subset of parameters.

.. ipython:: python

   from rateslib import IRS
   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="1Y",
       frequency="S",
       payment_lag=4,
       notional=50e6,
       amortization=10e6
   )
   irs.leg1.schedule.frequency
   irs.leg1.schedule.payment_lag
   irs.leg1.notional
   irs.leg1.amortization
   irs.leg2.schedule.frequency  # <- Inherited
   irs.leg2.schedule.payment_lag  # <- Inherited
   irs.leg2.notional  # <- Inherited with negate
   irs.leg2.amortization  # <- Inherited with negate


Defaults
********

The ``defaults`` object is a global instance of the :class:`~rateslib.default.Defaults` class.
Its purpose is to provide necessary values when a user does not supply inputs. In the above
swap the user provided no ``convention``, ``modifier`` or ``currency``. These have been set
by default.

.. ipython:: python

   irs.leg1.schedule.modifier
   irs.leg1.convention
   irs.leg1.currency

The defaults values can be seen by calling its :meth:`~rateslib.defaults.Defaults.print` method.

.. ipython:: python

   from rateslib import defaults
   print(defaults.print())

These values can also be set:

.. ipython:: python

   defaults.convention = "ACT365F"
   defaults.base_currency = "gbp"
   irs = IRS(effective=dt(2022, 1, 1), termination="1Y", frequency="A")
   irs.leg1.convention  # <- uses new default value
   irs.leg1.currency  # <- uses new default value

   defaults.reset_defaults()  # <- reverse the changes.
