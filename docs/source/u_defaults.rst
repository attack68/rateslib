.. _defaults-doc:

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

The easiest way to construct conventional instruments is to use the ``spec`` argument (detailed
below)

.. ipython:: python

   from rateslib import XCS, dt
   xcs = XCS(
       effective=dt(2022, 1, 1),
       termination="10Y",
       spec="gbpusd_xcs"
   )
   xcs.kwargs

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

Market conventions and the ``spec`` argument
--------------------------------------------

To provide maximal flexibility a number of market conventions have already been pre-added to
*rateslib*. For an *Instrument* that allows the ``spec`` (specification) argument a host of
arguments will be pre-populated. The list of instrument specifications defined can
be seen by printing as below.
Note that some of these are aliases, for example *"sofr"* and *"usd_irs"* are the same, as are
*"usd_gb"* and *"ust"*.

.. ipython:: python

   print(defaults.spec.keys())

The individual parameters of an instrument can then be seen, as below for an example USD SOFR IRS,
with:

.. ipython:: python

   defaults.spec["usd_irs"]

.. warning::

   When using the ``spec`` argument, arguments which might normally inherit might be defined
   specifically, and will no longer inherit. If overwriting an instrument that has been directly
   specified, ensure to overwrite both legs.

We can change the frequency on the XCS defined in the initial example. Since ``leg2_frequency``
was explicitly defined by the ``spec`` then it will no longer inherit.

.. ipython:: python

   xcs = XCS(
       effective=dt(2022, 1, 1),
       termination="10Y",
       frequency="S",
       spec="gbpusd_xcs",
   )  # `leg2_frequency` will NOT be inherited as "S", it will be "Q" as defined by the `spec`
   xcs.kwargs

Values that are shown here as *NoInput* are populated when the individual legs are
instantiated and the values will then be set by default. For example we have that,

.. ipython:: python

   xcs.leg1.schedule.roll
