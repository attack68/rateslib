.. _whatsnew-doc:

.. role:: red

**************
Release Notes
**************

The future development of *rateslib* is open to many avenues.
Some possibilities are listed below. The author is very interested in any feedback
and this can be given on the public **Issues** board at the project github
repository: `Rateslib Project <https://github.com/attack68/rateslib>`_, or by direct
email contact through **rateslib@gmail.com**.

1.2.0 (27th May 2024)
**********************

This version uses **Rust** bindings. See :ref:`getting started <pricing-doc>`
for notes about installation changes.

New *FX Volatility Products* are set to **beta** status, probably until version 2.0.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Performance
     - The modules ``rateslib.dual`` and ``rateslib.splines`` have been ported to **Rust**
       instead of Python to improve calculation times.
   * - Splines
     - New methods :meth:`~rateslib.splines.ppev_single_dual`,
       :meth:`~rateslib.splines.ppev_single_dual2`, :meth:`~rateslib.splines.ppdnev_single_dual`,
       and :meth:`~rateslib.splines.ppdnev_single_dual2` have been added to ensure correct
       handling of AD with regards to both x-axis and y-axis variables. See
       :ref:`section on using AD with splines <splines-ad-doc>`
   * - Splines
     - Added :meth:`~rateslib.splines.evaluate` for automatically handling which *ppdnev* method
       to use based on the AD sensitivities of the given `x` value.
   * - Instruments
     - :red:`Breaking Changes!` Amend :class:`~rateslib.instruments.FXExchange` to **remove** the
       arguments ``currency`` and ``leg2_currency``
       in favour of using ``pair`` which is consistent with the new *FX Volatility* naming convention.
       Also **reverse** the ``notional`` so that a +1mm EURUSD transaction is considered as a purchase of
       EUR and a sale of USD.
   * - Instruments
     - :class:`~rateslib.instruments.FXSwap` allows the dominant ``pair`` argument, consistent with other *FX*
       instruments to define the currencies. ``currency`` and ``leg2_currency`` are still currently permissible if
       ``pair`` is omitted.
   * - Instruments
     - Basic *FX Volatility Instruments* have been added in **beta** status, including
       :class:`~rateslib.instruments.FXCall`, :class:`~rateslib.instruments.FXPut`,
       :class:`~rateslib.instruments.FXRiskReversal`, :class:`~rateslib.instruments.FXStraddle`,
       :class:`~rateslib.instruments.FXStrangle`, :class:`~rateslib.instruments.FXBrokerFly`
       and :class:`~rateslib.instruments.FXOptionStrat`.
       See :ref:`user guide section <fx-volatility-doc>` for more information.
   * - FX Volatility
     - A new pricing component :class:`~rateslib.fx_volatility.FXDeltaVolSmile` has been added
       to allow pricing of single expiry *FX Options* with a *Smile* interpolated over a *Delta*
       axis.
   * - AD
     - Added :meth:`~rateslib.dual.dual_norm_pdf` for AD safe standard normal probability density.
   * - AD
     - Added :meth:`~rateslib.solver.newton_1dim` and :meth:`~rateslib.solver.newton_ndim`
       for AD safe Newton root solving in one or multiple dimensions.
   * - Solver
     - Added :meth:`~rateslib.solver.quadratic_eqn` to return the solution of a quadratic equation
       in an AD safe and consistent return format to other solvers for convenience.
   * - Bug
     - "ActActICMA" convention now handles ``frequency`` of "Z", asserting that of "A",
       albeit with a *UserWarning*.
   * - Bug
     - ``npv`` and ``cashflows`` of a :class:`~rateslib.periods.FloatPeriod` did not
       handle error messages regarding missing RFR fixings for a historical period.
       Calculations wll now raise if missing ``fixings``.
   * - Bug
     - `FXSwap` now no longer raises `TypeError` for dual number type mixing when `npv` or `rate`
       are called after changing the AD order of curves and fx objects.


1.1.0 (20th Mar 2024)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Automatic Differentiation
     - :red:`Breaking Change!` Dual number `gradient` method is no longer calculable on the object.
       Instead of `dual.gradient(vars)` use the following call `gradient(dual, vars)`, using the
       provided function :meth:`rateslib.dual.gradient`.
   * - Instruments
     - Added argument ``metric`` to :class:`~rateslib.instruments.Value` so that specific *Curve* values derived
       as calculated figures (e.g. continuously compounded zero rate, or index value) can be calibrated by *Solvers*.
   * - Bug
     - :meth:`~rateslib.solver.Solver.delta` and :meth:`~rateslib.solver.Solver.gamma` now work directly with
       given ``npv`` when ``fx`` is not provided.
   * - Bug
     - :meth:`~rateslib.periods.FloatPeriod.npv` now returns 0.0 for historical payment dates correctly when
       given the ``local`` argument.
   * - Bug
     - :meth:`~rateslib.periods.IndexCashflow.cashflows` no longer prints dual numbers to tables.
   * - Performance
     - Curve iterations in the :class:`~rateslib.solver.Solver` were amended in the way they handle
       :class:`~rateslib.dual.Dual` variables in order to reduce upcasting and increase the speed of basic operations.
   * - Performance
     - :class:`~rateslib.splines.bsplev_single` introduced a short circuit based on the positivity and support
       property to greatly improve time needed to solve curves with splines.
   * - Performance
     - :class:`~rateslib.curves.Curve` with splines are remapped to use float posix timestamps rather than datetimes
       for building splines. Operations with floats are much faster than their equivalents using timedeltas.


1.0.0 (1st Feb 2024)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Bug
     - :meth:`~rateslib.instruments.FRA.cashflows` now correctly identifies the DF at cash
       settled payment date.
   * - Bug
     - :meth:`~rateslib.legs.FloatLeg.fixings_table` now generates exact results (not in approximate mode) when RFR
       fixings are included in any period.


0.7.0 (29th Nov 2023)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Legs
     - Refactor how the ``defaults.fixings`` object works. **Breaking change**. Explained in
       :ref:`Working with Fixings <cook-fixings-doc>`.
   * - Legs
     - Allow ``fixings`` as a 2-tuple to manually define the first *FloatPeriod* (say as IBOR stub)
       and determine the rest from a *Series*. Also allow ``fx_fixings`` as a 2-tuple for similar
       reason for MTM *XCS*.
   * - Instruments
     - :class:`~rateslib.instruments.Fly` and :class:`~rateslib.instruments.Spread` now express
       *rate* in basis point terms and not percent.
   * - Instruments
     - Added ``calc_mode`` to :class:`~rateslib.instruments.BondFuture` to calculate CME US treasury
       conversion factors correctly.
   * - Instruments
     - :class:`~rateslib.instruments.BondFuture.ctd_index` can now optionally return the ordered set of CTD indexes
       instead of just the CTD.
   * - Instruments
     - Added :meth:`~rateslib.instruments.BondFuture.cms` to perform multi-security CTD analysis on
       :class:`~rateslib.instruments.BondFuture`.
   * - Solver
     - Add an attribute ``result`` that contains retrievable iteration success or failure
       information.
   * - Bug
     - Update :meth:`~rateslib.instruments.STIRFuture.analytic_delta` for
       :class:`~rateslib.instruments.STIRFuture` to match *delta*.
   * - Bug
     - Add the ``spec`` argument functionality missing for
       :class:`~rateslib.instruments.IndexFixedRateBond`.
   * - Bug
     - :class:`~rateslib.curves.CompositeCurve` now returns zero for DF item lookups prior to the initial node date.
   * - Bug
     - :class:`~rateslib.instruments.BondFuture.net_basis` now deducts accrued from the result when the prices are
       provided ``dirty``.

0.6.0 (19th Oct 2023)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Add a :class:`~rateslib.instruments.STIRFuture` class
   * - Instruments
     - Merge all :class:`~rateslib.instruments.XCS` classes into one, adding new arguments,
       ``fixed``, ``leg2_fixed`` and ``leg2_mtm`` to differentiate between types.
   * - Curves
     - Separate :class:`~rateslib.curves.MultiCsaCurve`
       from :class:`~rateslib.curves.CompositeCurve` for increased transparency on its action.
   * - Curves
     - Add the ability to supply curves in a dict for forecasting *FloatPeriods* to be
       able handle interpolated stub periods under an *"ibor"* ``fixing_method``.
   * - Solver
     - Added the methods :meth:`~rateslib.solver.Solver.jacobian` and
       :meth:`~rateslib.solver.Solver.market_movements` for coordinating multiple *Solvers*.
   * - Bug
     - Instrument ``spec`` with ``method_param`` set to 2 day lag for certain IBOR instruments.
   * - Bug
     - The :meth:`~rateslib.instruments.Portfolio.npv` method on a *Portfolio* no longer allows
       mixed currency outputs to be aggregated into a single float value.
   * - Bug
     - Now emit a warning if a discount factor or rate is requested on a curve with a spline
       outside of the rightmost boundary of the spline interval.


0.5.1 (11 Sep 2023)
**********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Rename :class:`~rateslib.instruments.FloatRateBond`
       to :class:`~rateslib.instruments.FloatRateNote` and removed the
       alias :class:`~rateslib.instruments.Swap`.
   * - Instruments
     - Add a ``spec`` keyword argument to allow instruments to be pre-defined and follow
       market conventions without the user needing to input these directly, but preserving an
       ability to overwrite specific values.
   * - Instruments
     - Add ``calc_mode`` to *Bonds* to provide mechanisms to perform YTM calculations under
       different conventions and geographies.
   * - Periods
     - :class:`~rateslib.periods.FloatPeriod` now allows **averaging** methods for
       determining the rate.
   * - Curves
     - The :meth:`shift()<rateslib.curves.Curve.shift>` operation for *Curves* now defaults to using
       a *CompositeCurve* approach to preserve a constant spread to the underlying *Curve* via
       a dynamic association. Shifted curves can also optionally add ``id`` and ``collateral``
       tags.
   * - Schedule
     - A :class:`~rateslib.scheduling.Schedule` now has the arguments ``eval_date`` and
       ``eval_mode`` allow a tenor-tenor effective-termination input.
   * - Defaults
     - Change the default :class:`~rateslib.solver.Solver` algorithm to *"levenberg_marquardt"*
       because it is more robust for new users, even if slower in general.
   * - Bug
     - :class:`~rateslib.instruments.FXExchange` can now be imported from *rateslib* and has been added
       to ``__init__``.
   * - Bug
     - :meth:`~rateslib.instruments.Sensitivities.cashflows_table` no longer returns empty when
       no collateral information is available.
   * - Bug
     - :meth:`~rateslib.periods.FloatPeriod.fixings_table` now properly represents published
       fixing values as having zero nominal exposure.
   * - Bug
     - ``solver.fx`` attribute is now properly passed through to the ``rate`` calculation
       of multi-currency instruments when ``fx`` is *None*.


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


0.3.1 (29 Jul 2023)
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

0.2.0 (15 May 2023)
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

0.1.0 (24 Apr 2023)
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