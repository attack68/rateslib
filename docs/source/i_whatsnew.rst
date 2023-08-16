.. _whatsnew-doc:

**************
Release Notes
**************

Rough Development Plan
***********************

The future development of *rateslib* is open to many avenues.
Some possibilities are listed below. The author is very interested in any feedback
and this can be given on the public **Issues** board at the project github
repository: `Rateslib Project <https://github.com/attack68/rateslib>`_, or by direct
email contact through **rateslib@gmail.com**.

.. list-table::
   :widths: 20 35 35 10
   :header-rows: 1


   * - Feature
     - Description
     - Consideration
     - Timeframe
   * - FX Spot and Repos
     - Adding basic funding instruments.
     - Likely (v1.0)
     - End 2023
   * - Coding Interest Rates
     - Officially document this library's algorithms and release the book.
     - Planned
     - End 2023
   * - Version 1.0
     - Release the official first non-beta version of this library.
     - Planned
     - End 2023
   * - Defaults
     - Adding the ability to define parameters by specification, e.g. "sofr irs" or
       "uk gilt", which set multiple default parameters.
     - Likely, to improve UI. (v1.0?)
     - By end 2023
   * - Vanilla FX options and volatility products
     - Adding option instruments and benchmark trades such as risk-reversals.
     - Highly likely (v2.0?)
     - By mid 2024
   * - Vanilla Swaptions
     - Adding the instruments priced by a volatility input.
     - Likely (v2.0 or v3.0?)
     - By end 2024
   * - SABR model for options
     - Adding the parameters to construct SABR vol surfaces/ cuves.
     - Possible, with dependencies to other developments. (v3.0?)
     - By end 2024
   * - Optimization of code
     - Using C extensions, or rust, or re-writing certain blocks to improve performance.
     - Likely to some degree, depending upon community adoption and contributions.
     - no ETA
   * - AD backend
     - Changing the AD implementation to another 3rd party (JAX, PyAudi)
     - Very unlikely, maturity of those libraries must increase and the performance
       improvements must be sufficient to warrant such a large codebase change.
     - no ETA


0.4.1 (not released)
**********************

New
---

- The ``shift`` method of curves now defaults to using a ``CompositeCurve`` approach to preserve
  association with the underlying curve. Shifted curves can also be assigned ``id`` and
  ``collateral`` tags.
- A ``Schedule`` can now be constructed with a tenor-tenor effective-termination input measured
  from some ``eval_date`` under an ``eval_mode`` construction methodology.

Bug Fixes
---------

- ``FXExchange`` can now be imported from ``rateslib`` and has been added to ``__init__``.
- ``cashflows_table`` no longer returns empty when no collateral information was available.
- ``fixings_table`` didn't properly represent published fixing values as zero exposure.

0.4.0 (12 Aug 2023)
********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Added ``split_notional`` to :class:`~rateslib.instruments.FXSwap` to more accurately
       reflect the interbank traded product.
   * - Instruments
     - Added :class:`~rateslib.instruments.FXExchange`, to provide booking FX spot or FX forward
       trades.
   * - Legs
     - Removed all ``LegExchange`` types, and replaced by adding ``initial_exchange`` and
       ``final_exchange`` as arguments to basic ``Legs``.
   * - Instruments
     - The ``payment_lag_exchange`` parameter for ``FXSwap`` was removed in favour of using
       ``payment_lag``.
   * - Defaults
     - Added historic fixing data until end July for ESTR, SOFR,
       SWESTR, SONIA and NOWA, for testing and validation.
   * - Instruments
     - Collateral tags were added to *Curves* to permit the new method ``cashflows_table`` which
       tabulates future cashflows according to currency and collateral type.
   * - Performance
     - Calendars are now cached which improves general performance by about 10%.
   * - Bug
     - When performing operations on *CompositeCurves* the resultant curve now correctly inherits
       the ``multi_csa`` parameters.
   * - Bug
     - ``FloatPeriod`` fixing exposure tables were marginally overestimated by ignoring
       discounting effects. This is corrected.
   * - Bug
     - NumPy.float128 datatype is not available on Windows and caused loading errors.
   * - Bug
     - The holiday calendars: 'ldn', 'tgt', 'nyc', 'stk', 'osl', and 'zur', have been reviewed
       and validated historic fixings against the historic fixing data. These are also now
       fully documented.
   * - Bug
     - *CompositeCurve* can now be constructed from *ProxyCurve* and *Curve* combinations.


v 0.3.0 (29 Jul 2023)
*********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Legs
     - Added :class:`~rateslib.legs.IndexFixedLeg`,
       :class:`~rateslib.legs.ZeroIndexLeg`,
       and :class:`~rateslib.legs.IndexFixedLegExchange`.
   * - Instruments
     - Added :class:`~rateslib.instruments.IndexFixedRateBond`,
       :class:`~rateslib.instruments.IIRS`, :class:`~rateslib.instruments.ZCIS`.
   * - Curves
     - Added :class:`~rateslib.curves.CompositeCurve`.

v 0.2.0 (15 May 2023)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Added :class:`~rateslib.instruments.BondFuture`.
   * - Curves
     - Added :class:`~rateslib.curves.IndexCurve`.

v 0.1.0 (24 Apr 2023)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Automatic Differentiation
     - A toolset for making risk sensitivity and gradient based calculations.
   * - Calendars
     - A toolset for handling dates and holiday calendars for schedules.
   * - Schedule
     - A toolset for generating financial schedules of financial instruments.
   * - Splines
     - A toolset for allowing spline interpolation.
   * - Curves
     - Initial classes for DF bases and value based interest rate curves.
   * - Periods
     - Initial classes for handling fixed periods, float periods and cashflows.
   * - Legs
     - Initial classes for aggregating periods.
   * - Instruments
     - Adding standard financial instruments such as securities: bonds and bills,
       and derivatives such as: IRS, SBS, FRA, XCS, FXSwap
   * - Solver
     - A set of algorithms for iteratively determining interest rate curves.
   * - FX
     - Initial classes for handling FX rates an Fx forwards.