.. _whatsnew-doc:

.. role:: red

**************
Release Notes
**************

The future development of *rateslib* is open to many avenues, see :ref:`the plan <developer-plan>`.
The author is very interested in any feedback
and this can be given on the public **Issues** board at the project github
repository: `Rateslib Project <https://github.com/attack68/rateslib>`_, or by direct
email contact, see `rateslib <https://rateslib.com>`_.

2.1.0 (Not released)
***************************

This release focused on restructuring curves in order to provide a system for user implemented
custom curves, which can directly inherit all of the native functionality of *rateslib*. An
example can be seen in the `Cookbook: Building Custom Curves (Nelson-Siegel) <z_basecurve.html>`_

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Pricing objects: Curves**
     - - :red:`Minor Breaking Change!`
         The classes :class:`~rateslib.curves._TranslatedCurve`,
         :class:`~rateslib.curves._RolledCurve` and
         :class:`~rateslib.curves._ShiftedCurve` are constructed as new objects
         to better handle the
         :meth:`~rateslib.curves._BaseCurve.translate`
         :meth:`~rateslib.curves._BaseCurve.roll`, and :meth:`~rateslib.curves._BaseCurve.shift`
         methods for curves.
         (`916 <https://github.com/attack68/rateslib/pull/916>`_)
         (`917 <https://github.com/attack68/rateslib/pull/917>`_)
         (`919 <https://github.com/attack68/rateslib/pull/919>`_)
       - :red:`Minor Breaking Change!`
         The ``composite`` argument for the :meth:`~rateslib.curves._BaseCurve.shift`
         method is removed, forcing shifted curves to always be dynamically dependent upon their
         underlying ``curve``.
         (`917 <https://github.com/attack68/rateslib/pull/917>`_)
       - :red:`Minor Breaking Change!` ``multi_csa_max_step`` and ``multi_csa_min_step`` are
         moved from the arguments of a :class:`~rateslib.curves.MultiCsaCurve` to the
         ``defaults`` object.
         (`922 <https://github.com/attack68/rateslib/pull/922>`_)
       - :red:`Minor Breaking Change!` The arguments ``calendar``, ``convention`` and ``modifier``
         are removed from a
         :class:`~rateslib.curves.ProxyCurve`. These meta items are inherited from the cashflow
         curve in the existing :class:`~rateslib.fx.FXForwards` object.
         (`925 <https://github.com/attack68/rateslib/pull/925>`_)
       - :red:`Minor Breaking Change!` A :class:`~rateslib.curves.MultiCsaCurve` no longer
         inherits a :class:`~rateslib.curves.CompositeCurve`, it forms a curve in its own right
         using the :class:`~rateslib.curves._BaseCurve` class.
         (`930 <https://github.com/attack68/rateslib/pull/930>`_)
       - A new :class:`~rateslib.curves.CreditImpliedCurve` class is available, in beta, to imply
         a *risk free*, *hazard*, or *credit* curve from combinations of the other two in
         combination with an expressed recovery rate.
         (`910 <https://github.com/attack68/rateslib/pull/910>`_)
       - The smile meta classes :class:`~rateslib.fx_volatility._FXDeltaVolSmileMeta` and
         :class:`~rateslib.fx_volatility._FXSabrSmileMeta` are consolidated into a single object
         :class:`~rateslib.fx_volatility._FXSmileMeta`.
         (`932 <https://github.com/attack68/rateslib/pull/932>`_)
   * - **Developers**
     - - (rust package) PyO3 0.23 -> 0.25  (`d-4 <https://github.com/attack68/rateslib-dev/pull/4>`_)
       - (rust package) Rust-Numpy 0.23 -> 0.25  (`d-4 <https://github.com/attack68/rateslib-dev/pull/4>`_)
       - (rust package) Bincode 1.3 -> 2.0  (`d-3 <https://github.com/attack68/rateslib-dev/pull/3>`_)


2.0.1 (10th June 2025)
***************************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Refactors**
     - - The ``expiries`` attribute is moved to ``meta`` on an
         :class:`~rateslib.fx_volatility.FXSabrSurface` to be consistent with an
         :class:`~rateslib.fx_volatility.FXDeltaVolSurface`.
         (`914 <https://github.com/attack68/rateslib/pull/914>`_)
   * - **Regressions**
     - - Flat *FXVolSurfaces*, parametrised by a **single** expiry and/or a **single** node value
         are now functional.
         (`913 <https://github.com/attack68/rateslib/pull/913>`_)
         (`915 <https://github.com/attack68/rateslib/pull/915>`_)

2.0.0 (4th June 2025)
*********************************

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/thumb_coding_2_1.png
         :alt: Coding Interest Rates: FX, Swaps and Bonds
         :target: https://www.amazon.com/dp/0995455562
         :width: 145
         :align: center

   .. container:: rightside60

      The publication to the left, *"Coding Interest Rates: FX, Swaps and Bonds 2"*
      documents the API architecture and mathematical algorithms for its objects
      upto and including the version two release of *rateslib*.

.. raw:: html

   <div class="clear" style="text-align: center; padding: 1em 0em 1em;"></div>

Some themes for this release involved:

- extensive revisions to use *indexes*.
  `Cookbook: Using Curves with an Index and Inflation Instruments <z_index_bonds_and_fixings.html>`_
  outlines best practice.
- extensions to bond calculation modes to provide more flexibility.
  `Cookbook: Understanding and Customising FixedRateBond Conventions <z_bond_conventions.html>`_
  outlines best practice.
- restructuring all pricing objects types' (:class:`~rateslib.curves.Curve`,
  :class:`~rateslib.curves.LineCurve`, :class:`~rateslib.fx_volatility.FXDeltaVolSmile`,
  :class:`~rateslib.fx_volatility.FXSabrSmile`) **attributes** to improve mutability safeguards,
  documentation and consistent type signatures. This also extends to pricing containers, such as
  (:class:`~rateslib.curves.ProxyCurve`, :class:`~rateslib.curves.CompositeCurve`,
  :class:`~rateslib.curves.MultiCsaCurve`, :class:`~rateslib.fx_volatility.FXDeltaVolSurface`,
  :class:`~rateslib.fx_volatility.FXSabrSurface`)

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Index Curves**, ``index_fixings`` **and** ``index_lag``
     - - :red:`Major Breaking Change!` The way ``index_fixings`` are treated when given as a *Series*
         now enforces that the data is provided with an ``index_lag`` of **zero** months, i.e.
         providing *actual* data. This is more convenient for handling *Instruments* with different
         ``index_lag`` and creates less functional risk. Calculations now allow *Curves*,
         *Instruments* and *Series* all to have different ``index_lag`` whilst ensuring correct
         calculations.
         (`807 <https://github.com/attack68/rateslib/pull/807>`_)
       - :red:`Minor Breaking Change!` The
         :meth:`Curve.index_value() <rateslib.curves.Curve.index_value>` method is changed to
         accept an ``index_lag`` argument which allows the determination of an *index value*
         for a specific date defined with a given *lag* and *interpolation* method. Also
         amended the way :class:`~rateslib.periods.IndexFixedPeriod` will handle the
         determination of cashflows given different ``index_lag`` specifications.
         (`802 <https://github.com/attack68/rateslib/pull/802>`_)
         (`803 <https://github.com/attack68/rateslib/pull/803>`_)
       - :red:`Minor Breaking Change!` ``index_fixings`` can  no longer be set as a *list* on *Legs*.
         Only a single value valid for the first period or a *Series* can be passed.
         (`807 <https://github.com/attack68/rateslib/pull/807>`_)
       - Add new method :meth:`~rateslib.curves.index_value` to determine an *index value* from a
         variety of sources including known fixings and/or a *Curve* if data from both those sources
         may need to be combined.
         (`809 <https://github.com/attack68/rateslib/pull/809>`_)
   * - **Bond Calculations & Conventions**
     - - :red:`Minor Breaking Change!` The argument names for
         :class:`~rateslib.instruments.BondCalcMode` are changed to
         drop the superfluous *'_type'* suffix.
         (`812 <https://github.com/attack68/rateslib/pull/812>`_)
       - Extend :class:`~rateslib.instruments.BondCalcMode` to support custom accrual,
         discount and cashflow functions for calculations. Italian BTP default, *'it_gb'*, is
         altered to now support delayed payments in the YTM formula.
         (`788 <https://github.com/attack68/rateslib/pull/788>`_)
         (`791 <https://github.com/attack68/rateslib/pull/791>`_)
         (`793 <https://github.com/attack68/rateslib/pull/793>`_)
         (`795 <https://github.com/attack68/rateslib/pull/795>`_)
         (`794 <https://github.com/attack68/rateslib/pull/794>`_)
       - Add bond calculation convention to support *'30U360'* accrued, and a new bond ``spec``
         *'us_corp'* and *'us_muni'* to support generic US corporate and municipal bonds.
         (`785 <https://github.com/attack68/rateslib/pull/785>`_)
         (`786 <https://github.com/attack68/rateslib/pull/786>`_)
         (`797 <https://github.com/attack68/rateslib/pull/797>`_)
       - The documentation page for the :class:`~rateslib.instruments.BondCalcMode` has been
         re-written to included all of the current formulae and structuring of bond accrual and
         yield-to-maturity calculations.
         (`790 <https://github.com/attack68/rateslib/pull/790>`_)
         (`789 <https://github.com/attack68/rateslib/pull/789>`_)
         (`794 <https://github.com/attack68/rateslib/pull/794>`_)
       - Add the ``spec`` *'ch_gb'* for Swiss government bonds and *'ch_gb_10y'* for EUREX
         10Y Swiss government bond futures along with the appropriate conversion factor
         calculations.
         (`834 <https://github.com/attack68/rateslib/pull/834>`_)
         (`835 <https://github.com/attack68/rateslib/pull/835>`_)
       - Add the initialisation argument ``metric`` to :class:`~rateslib.instruments.FixedRateBond`,
         :class:`~rateslib.instruments.IndexFixedRateBond`, :class:`~rateslib.instruments.Bill`,
         :class:`~rateslib.instruments.FloatRateNote`, for easier integration into a
         :class:`~rateslib.solver.Solver`, and for use with a :class:`~rateslib.instruments.Spread`,
         *Instrument*.
         (`845 <https://github.com/attack68/rateslib/pull/845>`_)
   * - **Calendars**
     - - Added a new method :meth:`~rateslib.calendars.next_imm` to determine the next IMM date
         from a given start date under different IMM methodologies.
         (`773 <https://github.com/attack68/rateslib/pull/773>`_)
       - Added a new day count convention *'30U360'* to :meth:`~rateslib.calendars.dcf`.
         (`780 <https://github.com/attack68/rateslib/pull/780>`_)
   * - **Pricing Objects: Curves, Smiles & Surfaces**
     - - :red:`Major Breaking Change!` The **attributes** associated with *Curves*, such as
         ``calendar``, ``convention``, ``collateral``, ``modifier``, ``index_base``, ``index_lag``
         ``nodes``, ``spline`` etc. have been migrated into data containers available as new
         **attributes** associated with any *Curve* type. In particular, see the objects:
         :class:`~rateslib.curves.utils._CurveMeta`,
         :class:`~rateslib.curves.utils._CurveInterpolator`,
         :class:`~rateslib.curves.utils._CurveNodes`,
         (`853 <https://github.com/attack68/rateslib/pull/853>`_)
         (`854 <https://github.com/attack68/rateslib/pull/854>`_)
         (`855 <https://github.com/attack68/rateslib/pull/855>`_)
         (`873 <https://github.com/attack68/rateslib/pull/873>`_)
       - :red:`Major Breaking Change!` The **attributes** associated with *FXVol* pricing objects
         are also organised into data containers available as new **attributes**. In particular,
         see the objects:
         :class:`~rateslib.fx_volatility.utils._FXDeltaVolSmileNodes`
         :class:`~rateslib.fx_volatility.utils._FXDeltaVolSmileMeta`
         :class:`~rateslib.fx_volatility.utils._FXDeltaVolSurfaceMeta`
         :class:`~rateslib.fx_volatility.utils._FXSabrSmileNodes`
         :class:`~rateslib.fx_volatility.utils._FXSabrSmileMeta`
         :class:`~rateslib.fx_volatility.utils._FXSabrSurfaceMeta`
         (`867 <https://github.com/attack68/rateslib/pull/867>`_)
         (`869 <https://github.com/attack68/rateslib/pull/869>`_)
         (`871 <https://github.com/attack68/rateslib/pull/871>`_)
         (`872 <https://github.com/attack68/rateslib/pull/872>`_)
         (`880 <https://github.com/attack68/rateslib/pull/880>`_)
         (`881 <https://github.com/attack68/rateslib/pull/881>`_)
         (`882 <https://github.com/attack68/rateslib/pull/882>`_)
       - :red:`Minor Breaking Change!` Additional **attributes** of a
         :class:`~rateslib.curves.ProxyCurve`
         have been restructured into a :class:`~rateslib.curves.utils._ProxyCurveInterpolator`
         class, to be consistent with the other attribute changes on *Curves*.
         (`900 <https://github.com/attack68/rateslib/pull/900>`_)
       - The *'linear'* and *'log_linear'* ``interpolation`` methods of a *Curve* now automatically
         adjust to business day interpolation when using a *'bus252'* ``convention``.
         (`821 <https://github.com/attack68/rateslib/pull/821>`_)
       - The attributes ``credit_discretization`` and ``credit_recovery_rate`` are
         added to the ``meta`` of a :class:`~rateslib.curves.Curve` to replace the **removed**,
         equivalent arguments of a
         :class:`~rateslib.periods.CreditProtectionPeriod`.
       - Add :meth:`~rateslib.curves.Curve.update_meta` method to update values of *Curve* meta
         data.
         (`887 <https://github.com/attack68/rateslib/pull/887>`_)
       - :red:`Minor Breaking Change!` The default ``index_lag`` for a
         :class:``~rateslib.curves.Curve` is set to zero.
         See the default setting ``index_lag_curve``.
         (`821 <https://github.com/attack68/rateslib/pull/821>`_)
       - :class:`~rateslib.curves.CompositeCurve` can now be constructed
         from other *CompositeCurves*.
         (`826 <https://github.com/attack68/rateslib/pull/826>`_)
       - The :meth:`Curve.shift() <rateslib.curves.Curve.shift>` method has its ``composite``
         argument moved in the signature and the calculation to determine shifted *Curves* is now
         more precise, albeit may impact slight performance degradations in bond OAS spread
         calculations.
         (`828 <https://github.com/attack68/rateslib/pull/828>`_)
         (`849 <https://github.com/attack68/rateslib/pull/849>`_)
       - The :meth:`~rateslib.curves.average_rate` method now requires a ``dcf`` input.
         (`836 <https://github.com/attack68/rateslib/pull/836>`_)
       - The caching of values of a :class:`~rateslib.curves.MultiCsaCurve` is improved and
         extended (`842 <https://github.com/attack68/rateslib/pull/842>`_)
       - Simple spline interpolation can now be automatically constructed by specifying
         *"spline"* as the argument for ``interpolation``. See docs.
         (`847 <https://github.com/attack68/rateslib/pull/847>`_)
       - :red:`Minor Breaking Change!` The argument ``c`` for spline coefficients is no longer
         available in the initialisation of a *Curve* class. This value is determined
         automatically to maintain consistency between supplied node values and solved spline
         coefficients.
         (`859 <https://github.com/attack68/rateslib/pull/859>`_)
       - :red:`Minor Breaking Change!` The arguments ``interpolation`` and ``endpoints`` are
         removed from the :meth:`Curve.update() <rateslib.curves.Curve.update>` method to
         avoid unnecessarily complicated mutations. Create new *Curve* instances instead.
         (`859 <https://github.com/attack68/rateslib/pull/859>`_)
       - The method :meth:`~rateslib.fx_volatility.FXDeltaVolSmile.csolve` is removed due to
         never being required to be called by a user directly.
         (`872 <https://github.com/attack68/rateslib/pull/872>`_)
       - A :class:`~rateslib.curves.ProxyCurve` is now returned from a cached object attributed
         to an :class:`~rateslib.fx.FXForwards` class and not as an isolated object instance,
         when calling :meth:`FXForwards.curve() <rateslib.fx.FXForwards.curve>`.
         (`899 <https://github.com/attack68/rateslib/pull/899>`_)
   * - **Automatic Differentiation & Algorithms**
     - - Operator overloads added to allow dual number exponents, i.e. :math:`z^p`, where *z*,
         *p* are dual number types. This facilitates AD for the SABR function as well as other
         exotic functions.
         (`767 <https://github.com/attack68/rateslib/pull/767>`_)
         (`768 <https://github.com/attack68/rateslib/pull/768>`_)
         (`769 <https://github.com/attack68/rateslib/pull/769>`_)
       - Implement a new type of iterative root solver, :meth:`~rateslib.dual.ift_1dim`, that
         solves a one-dimensional implicit function if its derivatives are not known but its inverse
         function is analytical.
         (`775 <https://github.com/attack68/rateslib/pull/775>`_)
         (`776 <https://github.com/attack68/rateslib/pull/776>`_)
         (`777 <https://github.com/attack68/rateslib/pull/777>`_)
         (`778 <https://github.com/attack68/rateslib/pull/778>`_)
   * - **Performance**
     - - Amend the iterative algorithm for YTM to widen the consecutive ytm search
         interval, but require one function evaluation per iteration instead of two, and use
         analytical formula instead of NumPy solve.
         (`781 <https://github.com/attack68/rateslib/pull/781>`_)
         (`782 <https://github.com/attack68/rateslib/pull/782>`_)
         (`783 <https://github.com/attack68/rateslib/pull/783>`_)
       - Modify the :meth:`CompositeCurve.rate() <rateslib.curves.CompositeCurve.rate>` method
         to use cached discount factors when compositing *Curve* types.
         This particularly improves performance for dual type calculations.
         (`816 <https://github.com/attack68/rateslib/pull/816>`_)
   * - **Serialization**
     - - Python wrapped Rust objects are now serialised with the identifier *'PyWrapped'* to
         distinguish between serialised, native Python objects which use the *'PyNative'*
         identifier. The *NoInput* type is also now handled in serialization of objects.
         (`855 <https://github.com/attack68/rateslib/pull/855>`_)
       - :red:`Major Breaking Change!` JSON serialization of :class:`~rateslib.curves.Curve` and
         :class:`~rateslib.curves.LineCurve` is refactored to suit the modification of the
         new *Curve* attributes structure.
         (`860 <https://github.com/attack68/rateslib/pull/860>`_)
   * - **Bug Fixes**
     - - The SABR functions are modified to handle ``expiry`` for an interpolated
         :class:`~rateslib.fx_volatility.FXSabrSurface`. Previously, the specific expiry was used to
         evaluate the volatility on each *SabrSmile*. Now the relevant *Smile* expiry is used as the
         entry to the SABR function before interpolating for the given expiry.
         (`757 <https://github.com/attack68/rateslib/pull/757>`_)
       - ``index_lag`` is now correctly passed to *Index* type *Period* construction during a
         *Leg* initialization.
         (`808 <https://github.com/attack68/rateslib/pull/808>`_)
       - Scalars on the different ``metrics`` for a :class:`~rateslib.instruments.Value` are
         amended to better reflect the unit derivatives in *delta* and *gamma* calculations
         (`806 <https://github.com/attack68/rateslib/pull/806>`_)
       - Add discount factor scaling to separate the difference of *'spot'* versus *'forward'*
         **sticky delta** calculation in *FXOption* greeks.
         (`792 <https://github.com/attack68/rateslib/pull/792>`_)
       - Add :class:`~rateslib.instruments.BondCalcMode` and
         :class:`~rateslib.instruments.BillCalcMode` to global *rateslib* namespace.
         (`812 <https://github.com/attack68/rateslib/pull/812>`_)
       - For *Curve* rate calculations the *Curve* ``calendar`` is now correctly passed to
         the :meth:`~rateslib.calendars.dcf` method for day count fraction determination.
         For almost all conventions this has no effect, but for "bus252", used in
         Brazil, for example, the right number of business days is essential to the
         calculation.
         (`817 <https://github.com/attack68/rateslib/pull/817>`_)
       - The AD order of a :class:`~rateslib.curves.CompositeCurve` is now determined from the
         maximum AD order of its contained *Curves* and no longer the first *Curve* supplied.
         (`829 <https://github.com/attack68/rateslib/pull/829>`_)
       - The :meth:`FXDeltaVolSmile.update <rateslib.fx_volatility.FXDeltaVolSmile.update>`
         method now updates the spline interpolator after a *nodes* update.
         (`844 <https://github.com/attack68/rateslib/pull/844>`_)
   * - **Deprecations & Removals**
     - - :red:`Major Breaking Change!` The arguments ``recovery_rate`` and ``discretization`` are
         removed from the :class:`~rateslib.periods.CreditProtectionPeriod`, and the associated
         downstream objects :class:`~rateslib.legs.CreditProtectionLeg`,
         :class:`~rateslib.instruments.CDS`.
         (`885 <https://github.com/attack68/rateslib/pull/885>`_)
         (`889 <https://github.com/attack68/rateslib/pull/889>`_)
         (`890 <https://github.com/attack68/rateslib/pull/890>`_)
       - :red:`Major Breaking Change!` The method :meth:`~rateslib.curves.interpolate` is
         removed and user defined callables provided to a Curve ``interpolation`` method adopt a
         new signature. Please review appropriate documentation and examples.
         (`820 <https://github.com/attack68/rateslib/pull/820>`_)
       - :red:`Minor Breaking Change!` The ``approximate`` argument is removed from the
         :meth:`CompositeCurve.rate() <rateslib.curves.CompositeCurve.rate>` method to create a
         more consistent *Curve* definition between O/N rates and discount factors and which is
         more performant.
         (`816 <https://github.com/attack68/rateslib/pull/816>`_)
       - All of the older default ``spec`` **aliases** have been removed. There is now only a
         single version of a particular ``spec``, which is as described in documentation.
         (`892 <https://github.com/attack68/rateslib/pull/892>`_)

1.8.0 (22nd April 2025)
****************************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Removed**
     - ``IndexCurve``, which was deprecated with warnings in 1.7.0, has been removed.
       (`691 <https://github.com/attack68/rateslib/pull/691>`_)
   * - Period
     - Add :class:`~rateslib.periods.NonDeliverableFixedPeriod`
       (`681 <https://github.com/attack68/rateslib/pull/681>`_)
   * - Calendars
     - Allow custom calendar additions to ``defaults.calendars`` and fast fetching with
       :meth:`~rateslib.calendars.get_calendar`.
       (`684 <https://github.com/attack68/rateslib/pull/684>`_)
   * - Instruments
     - Add ``calc_mode`` *'eurex_eur'* for :class:`~rateslib.instruments.BondFuture`.
       (`699 <https://github.com/attack68/rateslib/pull/699>`_)
   * - Instruments
     - Add ``spec`` argument for :class:`~rateslib.instruments.BondFuture`, and some CME treasury futures and EUREX
       bond future default specifications. This has also refactored the *BondFuture* attributes into a ``kwargs``
       dict instead of being directly accessible on the object. This may affect existing code that relies on these
       attributes.
       (`700 <https://github.com/attack68/rateslib/pull/700>`_)
   * - Instruments
     - Add **sticky delta** calculation output to
       :meth:`FXOption.analytic_greeks <rateslib.instruments.FXOption.analytic_greeks>`
       (`749 <https://github.com/attack68/rateslib/pull/749>`_)
   * - FX Volatility
     - An :class:`~rateslib.fx_volatility.FXSabrSmile` is implemented in *beta* status.
       (`714 <https://github.com/attack68/rateslib/pull/714>`_)
   * - FX Volatility
     - An :class:`~rateslib.fx_volatility.FXSabrSurface` is implemented in *beta* status.
       (`729 <https://github.com/attack68/rateslib/pull/729>`_)
   * - FX Volatility
     - :red:`Minor Breaking Change!` The arguments to all FX Volatility model objects'
       :meth:`~rateslib.fx_volatility.FXDeltaVolSmile.get_from_strike` methods are reordered
       to prioritise ``expiry`` which is more commonly required for *Surfaces*.
       (`735 <https://github.com/attack68/rateslib/pull/735>`_)
   * - Performance
     - The :meth:`FXStrangle.rate <rateslib.instruments.FXStrangle.rate>` method is refactored to
       use :meth:`rateslib.dual.newton_1dim` for performance.
       (`738 <https://github.com/attack68/rateslib/pull/738>`_)
   * - Performance
     - A cache has been added to :class:`~rateslib.fx.FXForwards` for forward FX rate caching
       per currency pair per date.
       (`761 <https://github.com/attack68/rateslib/pull/761>`_)
   * - Refactor
     - All pricing objects, such as :class:`~rateslib.curves.Curve`, :class:`~rateslib.fx.FXRates`,
       :class:`~rateslib.fx_volatility.FXDeltaVolSmile` etc., and pricing containers, such as
       :class:`~rateslib.curves.CompositeCurve`, :class:`~rateslib.fx.FXForwards`,
       :class:`~rateslib.fx_volatility.FXDeltaVolSurface` etc., have moved their AD identifying
       attribute to the private value ``_ad`` instead of ``ad``, although ``ad`` is still readable.
       (`738 <https://github.com/attack68/rateslib/pull/738>`_)
   * - Refactor
     - Rename :class:`~rateslib.instruments.BaseMixin` to :class:`~rateslib.instruments.Metrics`.
       (`678 <https://github.com/attack68/rateslib/pull/678>`_)
   * - Refactor
     - Minor changes to :class:`BondFuture.cms <rateslib.instruments.BondFuture.cms>` to avoid
       the proceeds method of repo rates and utilise only a bond curve for forward bond prices.
       (`693 <https://github.com/attack68/rateslib/pull/693>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` The argument ``notional`` in
       :class:`~rateslib.instruments.NDF` now **always** refers to the *reference currency* and
       **never** the *settlement currency*. The :meth:`~rateslib.instruments.NDF.cashflows` method
       is also now more explicit and shows both the settlement exchange and the converted amount
       of the deliverable cashflow.
       (`695 <https://github.com/attack68/rateslib/pull/695>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` The argument ``reference_currency`` is renamed ``currency``,
       and the argument ``settlement`` is renamed ``payment`` in
       :class:`~rateslib.periods.NonDeliverableCashflow`.
       (`677 <https://github.com/attack68/rateslib/pull/677>`_)
       (`694 <https://github.com/attack68/rateslib/pull/694>`_)
   * - Bug
     - :meth:`FXDeltaVolSmile.get <rateslib.fx_volatility.FXDeltaVolSmile.get>` fixes a bug
       where the delta index was not properly generated for ``delta_types`` with different
       premium adjustments. :red:`Minor Breaking Change!` Also changes the arguments to the
       method to make it more user friendly, removing ``w_deli`` and ``w_spot`` and using a
       single value ``z_w`` which is the quotient of the previous two.
       (`742 <https://github.com/attack68/rateslib/pull/742>`_)
   * - Bug
     - Add :class:`~rateslib.instruments.NDF` to global *rateslib* namespace.
       (`682 <https://github.com/attack68/rateslib/pull/682>`_)
   * - Bug
     - Add :class:`~rateslib.legs.CreditProtectionLeg`,
       :class:`~rateslib.legs.CreditPremiumLeg`, :class:`~rateslib.periods.CreditProtectionPeriod`,
       :class:`~rateslib.periods.CreditPremiumPeriod` and
       :class:`~rateslib.periods.NonDeliverableCashflow` to global *rateslib* namespace.
       (`697 <https://github.com/attack68/rateslib/pull/697>`_)
   * - Bug
     - The ``fx_rates_immediate`` attribute on the :class:`~rateslib.fx.FXForwards` class now
       preserves AD sensitivity to the initial discount factor on the ``fx_curves``. Although this
       is assumed to be, constantly, 1.0 and has no effect on risk sensitivity calculations
       it is more consistent for unit test building.
       (`712 <https://github.com/attack68/rateslib/pull/712>`_)
   * - Bug
     - Correct an issue where *Solver* dependency chains were incorrectly constructed in the case
       of mismatching numbers of *Curve* variables and calibrating *Instruments* leading to
       *ValueErrors* for *delta* and *gamma* calculations.
       (`744 <https://github.com/attack68/rateslib/pull/744>`_)

1.7.0 (31st January 2025)
****************************

The key theme for 1.7.0 was to add Python type hinting to the entire codebase, and adding
``mypy`` CI checks to the development process. This resulted in
a number of refactorisations which may have changed the way some argument inputs should be
structured.

*FXOptions* which were added and listed in beta status since v1.2.0, have seen the largest
changes and have now been moved out beta status.

Internally, caching and state management were improved to provide more safety, preventing users
inadvertently mutating objects without the *Solver's* *Gradients* being updated. All mutable
objects now have specific methods to allow *updates*.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - **Deprecation**
     - :class:`~rateslib.curves.IndexCurve` is deprecated. Use :class:`~rateslib.curves.Curve`
       instead.
       (`560 <https://github.com/attack68/rateslib/pull/560>`_)
   * - Instruments
     - :meth:`~rateslib.instruments.FloatRateNote.ytm` added to
       :class:`~rateslib.instruments.FloatRateNote` to allow the calculation of
       yield-to-maturity for that *Instrument* based on ``calc_mode`` similar to
       *FixedRateBonds*. (`529 <https://github.com/attack68/rateslib/pull/529>`_)
   * - Instruments
     - :class:`~rateslib.periods.NonDeliverableCashflow` and
       :class:`~rateslib.instruments.NDF` added to allow FX forwards settled in
       an alternate currency to be valued.
       (`647 <https://github.com/attack68/rateslib/pull/647>`_)
       (`651 <https://github.com/attack68/rateslib/pull/651>`_)
   * - Instruments
     - Add parameter ``expiry`` to :class:`~rateslib.instruments.VolValue` to permit more
       flexibility in calibrating *FXDeltaVolSurfaces*.
       (`658 <https://github.com/attack68/rateslib/pull/658>`_)
   * - Splines
     - The *Spline* :meth:`~rateslib.splines.evaluate` method is enhanced to allow an x-axis
       evaluation if a :class:`~rateslib.dual.Variable` is passed, through dynamic *Dual* or *Dual2*
       conversion.
       (`558 <https://github.com/attack68/rateslib/pull/558>`_)
   * - Curves
     - Add methods :meth:`~rateslib.curves.Curve.update` and
       :meth:`~rateslib.curves.Curve.update_node` to allow mutating *Curve* types directly
       with appropriate cache and state management.
       (`584 <https://github.com/attack68/rateslib/pull/584>`_)
   * - Curves
     - Caching and state management was extended to :class:`~rateslib.curves.MultiCsaCurve` and
       the *defaults* option ``curve_caching_max`` (initially set to 1000 elements) was added
       to prevent memory issues of unlimitedly expanding caches.
       (`661 <https://github.com/attack68/rateslib/pull/661>`_)
   * - Calendars
     - Add *"mum"* (INR: Mumbai) to list of default calendars.
       (`659 <https://github.com/attack68/rateslib/pull/659>`_)
   * - Bug
     - Defaults spec *"usd_stir1"* for CME 1m SOFR futures, and *"eur_stir1"* for ICE 1m ESTR
       futures has corrected the
       ``roll`` to *"som"*, instead of *"imm"*, to allow correct placement of contracts averaging
       all of the rates in a specific contract month.
       (`631 <https://github.com/attack68/rateslib/pull/631>`_)
   * - Bug
     - :class:`~rateslib.instruments.STIRFuture` now correctly handles the ``fx`` and ``base``
       arguments when using the :meth:`~rateslib.instruments.STIRFuture.npv` or
       :meth:`~rateslib.instruments.STIRFuture.analytic_delta` methods.
       (`519 <https://github.com/attack68/rateslib/pull/519>`_)
   * - Bug
     - :class:`~rateslib.instruments.STIRFuture` now correctly handles *NPV* when ``fx``
       is provided as an, potentially unused, argument.
       (`653 <https://github.com/attack68/rateslib/pull/653>`_)
   * - Bug
     - :class:`~rateslib.fx.FXForwards` corrects a bug which possibly mis-ordered some
       currencies if a ``base`` argument was given at initialisation, yielding mis-stated FX rates
       for some pair combinations.
       (`669 <https://github.com/attack68/rateslib/pull/669>`_)
   * - Bug
     - :meth:`~rateslib.periods.FloatPeriod.rate` now correctly calculates when ``fixings``
       are provided in any of the acceptable formats and contains all data to do so, in the
       absense of a forecast ``curve``, instead of returning *None* for some cases.
       This allows for :meth:`~rateslib.periods.FloatPeriod.cashflows` to return values even
       when ``curve`` is not constructed.
       (`530 <https://github.com/attack68/rateslib/pull/530>`_)
       (`532 <https://github.com/attack68/rateslib/pull/532>`_)
       (`535 <https://github.com/attack68/rateslib/pull/535>`_)
       (`536 <https://github.com/attack68/rateslib/pull/536>`_)
   * - Bug
     - :meth:`~rateslib.legs.CustomLeg` now allows construction from recently constructed
       *Period* types including *CreditProtectionPeriod*, *CreditPremiumPeriod*,
       *IndexCashflow* and *IndexFixedPeriod*.
       (`596 <https://github.com/attack68/rateslib/pull/596>`_)
   * - Dependencies
     - Drop support for Python 3.9, only versions 3.10 - 3.13 now supported.
   * - Refactor
     - :class:`~rateslib.curves.CompositeCurve` no longer requires all curves to have the same ``index_base``
       or ``index_lag``. Those values will be sampled from the first provided composited *Curve*.
   * - Refactor
     - The builtin ``abs`` method operating on dual type objects now returns dual type objects with properly
       adjusted dual manifold gradients. The previous functionality returning only floats can be replicated
       using the internal method :meth:`rateslib.dual._abs_float`.
   * - Refactor
     - :red:`Minor Breaking Change!` :meth:`~rateslib.calendars.get_calendar` has dropped the
       ``kind`` argument being only useful internally.
       (`524 <https://github.com/attack68/rateslib/pull/524>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` :meth:`FXForwards.rate <rateslib.fx.FXForwards.rate>`
       has dropped the ``path`` and ``return_path`` arguments being mainly useful internally.
       Replicable functionality is achieved by importing and using the internal method
       :meth:`rateslib.fx.FXForwards._rate_with_path`.
       (`537 <https://github.com/attack68/rateslib/pull/537>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` :meth:`FXForwards.update <rateslib.fx.FXForwards.update>`
       has dropped the ``fx_curves`` argument and amended the ``fx_rates`` argument to
       provide a safer architecture for mutability of objects after market data changes.
       (`544 <https://github.com/attack68/rateslib/pull/544>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` :meth:`Curve.to_json <rateslib.curves.Curve.to_json>`
       has refactored its JSON format to include the Rust calendar serialization implementations
       introduced in v1.3.0. This should not be noticeable on round trips, i.e. using
       ``from_json`` on the output from ``to_json``.
       (`552 <https://github.com/attack68/rateslib/pull/552>`_)
   * - Refactor
     - Internal ``_cache_id`` management is introduced to mutable objects such as *Curves*,
       *FXRates* and *FXForwards* to allow auto-mutate detection of associated objects and ensure
       consistent method results.
       (`570 <https://github.com/attack68/rateslib/pull/570>`_)
   * - Refactor
     - The internal data objects for *FXOption* pricing are restructured to conform to more
       strict data typing.
       (`642 <https://github.com/attack68/rateslib/pull/642>`_)
   * - Refactor
     - :red:`Minor Breaking Change!` The argument inputs for *FXOptionStrat* types, such
       as :class:`~rateslib.instruments.FXRiskReversal`, :class:`~rateslib.instruments.FXStraddle`,
       :class:`~rateslib.instruments.FXStrangle` and :class:`~rateslib.instruments.FXBrokerFly`,
       may have changed to conform to a more generalised structure. This may include the
       specification of their ``premium``, ``strike``, ``notional`` and ``vol`` inputs. Review
       their updated documentation for details.
       (Mostly `643 <https://github.com/attack68/rateslib/pull/643>`_)
   * - Developers
     - *rateslib-rs* extension upgrades to using PyO3:0.23, numpy:0.23, itertools:0.14,
       statrs:0.18, indexmap:2.7
       (`655 <https://github.com/attack68/rateslib/pull/655>`_)
       (`656 <https://github.com/attack68/rateslib/pull/656>`_)

1.6.0 (30th November 2024)
****************************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Add :class:`~rateslib.instruments.CDS` for credit pricing, as well as the associated components;
       :class:`~rateslib.legs.CreditPremiumLeg`, :class:`~rateslib.periods.CreditPremiumPeriod`,
       :class:`~rateslib.legs.CreditProtectionLeg`, :class:`~rateslib.periods.CreditProtectionPeriod`.
       (`419 <https://github.com/attack68/rateslib/pull/419>`_)
       (`425 <https://github.com/attack68/rateslib/pull/425>`_)
       (`426 <https://github.com/attack68/rateslib/pull/426>`_)
   * - Instruments
     - Add an additional method :meth:`~rateslib.instruments.CDS.analytic_rec_risk` to measure the
       sensitivity of a change in ``recovery_rate`` for a :class:`~rateslib.instruments.CDS`.
       (`448 <https://github.com/attack68/rateslib/pull/448>`_)
   * - Instruments
     - Add the ``spec`` options; *'audusd_xcs'*, *'audusd_xcs3'*, *'nzdusd_xcs3'*, *'nzdaud_xcs3'*,
       *'us_ig_cds'*
       (`429 <https://github.com/attack68/rateslib/pull/429>`_)
       (`454 <https://github.com/attack68/rateslib/pull/454>`_)
   * - Instruments
     - Add a :meth:`~rateslib.instruments.IRS.fixings_table` method to floating rate based
       *Instruments*: *IRS*, *SBS*, *FRA*, *IIRS*, *ZCS*, *STIRFuture*, *FloatRateNote*.
       (`467 <https://github.com/attack68/rateslib/pull/467>`_)
       (`470 <https://github.com/attack68/rateslib/pull/470>`_)
       (`490 <https://github.com/attack68/rateslib/pull/490>`_)
       (`493 <https://github.com/attack68/rateslib/pull/493>`_)
       (`499 <https://github.com/attack68/rateslib/pull/499>`_)
       (`500 <https://github.com/attack68/rateslib/pull/500>`_)
       (`510 <https://github.com/attack68/rateslib/pull/510>`_)
   * - Instruments
     - Add a :meth:`~rateslib.instruments.Portfolio.fixings_table` method to *Portfolio*, *Fly*,
       *Spread* to aggregate fixings tables on contained and applicable *Instruments*.
       (`491 <https://github.com/attack68/rateslib/pull/491>`_)
       (`508 <https://github.com/attack68/rateslib/pull/508>`_)
   * - Legs
     - Add method :meth:`~rateslib.legs.FloatLegMtm.fixings_table` to a *FloatLegMtm* and
       *ZeroFloatLeg*.
       (`480 <https://github.com/attack68/rateslib/pull/480>`_)
       (`482 <https://github.com/attack68/rateslib/pull/482>`_)
       (`489 <https://github.com/attack68/rateslib/pull/489>`_)
   * - Periods
     - :red:`Minor Breaking Change!` The method :meth:`~rateslib.periods.FloatPeriod.fixings_table`
       returns a *DataFrame* with amended column headers to reference the *Curve* id from which
       the fixing notionals are derived, and populates additional columns.
   * - Performance
     - *Curve caching* introduced to :class:`~rateslib.curves.Curve`, :class:`~rateslib.curves.LineCurve`,
       :class:`~rateslib.curves.IndexCurve` to improve performance of repeatedly fetched curve values such as
       in *Solvers* and standardised *Instruments*. This feature can be opted out of using the
       ``defaults.curve_caching`` setting. Note also the added :meth:`~rateslib.curves.Curve.clear_cache` method.
       (`435 <https://github.com/attack68/rateslib/pull/435>`_)
   * - Performance
     - *Smile caching* introduced to :class:`~rateslib.fx_volatility.FXDeltaVolSurface`,
       to improve performance of fetched *Smiles* at repeated ``expiries``.
       This feature can be opted out of using the
       ``defaults.curve_caching`` setting.
       Note also the added :meth:`~rateslib.fx_volatility.FXDeltaVolSurface.clear_cache` method.
       (`481 <https://github.com/attack68/rateslib/pull/481>`_)
   * - Automatic Differentiation
     - Add a new object for AD management, a :class:`~rateslib.dual.Variable`, which allows a
       user to inject manual exogenous sensitivities into calculations. See
       :ref:`what is an exogenous Variable? <cook-exogenous-doc>`
       (`452 <https://github.com/attack68/rateslib/pull/452>`_)
   * - Risk Sensitivities
     - Add method :meth:`~rateslib.instruments.Sensitivities.exo_delta` to calculate the delta
       sensitivity against a user-defined exogenous *Variable*.
       (`453 <https://github.com/attack68/rateslib/pull/453>`_)
   * - Dependencies
     - **Python 3.13** *(with GIL)* is officially supported and tested.
       (`463 <https://github.com/attack68/rateslib/pull/463>`_)
   * - Bug
     - :class:`~rateslib.curves.MultiCsaCurve` and :class:`~rateslib.calendars.get_imm` are now
       included in the main namespace.
       (`436 <https://github.com/attack68/rateslib/pull/436>`_)
       (`486 <https://github.com/attack68/rateslib/pull/486>`_)
   * - Bug
     - Adding *Dual* or *Dual2* type ``spread`` using :meth:`~rateslib.curves.Curve.shift` method
       now avoids *TypeErrors* where possible and maintains appropriate AD orders for each
       existing and new object.
       (`440 <https://github.com/attack68/rateslib/pull/440>`_)
   * - Bug
     - The method :meth:`~rateslib.periods.FloatPeriod.fixings_table` is amended for IBOR type
       fixings to account for DCFs, amended payment dates, and interpolated stubs. Requires
       a new ``disc_curve`` argument for proper discounting.
       (`470 <https://github.com/attack68/rateslib/pull/470>`_)
   * - Bug
     - No longer allow the creation of very short *Schedules* with holiday dates that
       collapse to empty *Periods*.
       (`484 <https://github.com/attack68/rateslib/pull/484>`_)
   * - Developers
     - *rateslib-rs* extension upgrades to using PyO3:0.22, nadarray:0.16, numpy:0.22.
       (`460 <https://github.com/attack68/rateslib/pull/460>`_)

1.5.0 (25th September 2024)
****************************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - Added *"nzd_irs3"*, *"nzd_irs6"*, *"se_gbb"* and *"uk_gbb"* to available ``spec`` defaults.
       (`397 <https://github.com/attack68/rateslib/pull/397>`_)
       (`403 <https://github.com/attack68/rateslib/pull/403>`_)
   * - Instruments
     - :class:`~rateslib.instruments.BondCalcMode` and :class:`~rateslib.instruments.BillCalcMode`
       added to allow more flexibility when adding new bond specifications with other
       defined calculation conventions.
       (`402 <https://github.com/attack68/rateslib/pull/402>`_)
   * - Calendars
     - Add a *"wlg"* calendar for New Zealand *IRS*.
       (`363 <https://github.com/attack68/rateslib/pull/363>`_)
   * - Calendars
     - Add a method, :meth:`~rateslib.calendars.get_imm`, to calculate IMM dates.
       `(371) <https://github.com/attack68/rateslib/pull/371>`_
   * - Serialization
     - *PPSplines* are now serializable. Read more :ref:`here <serialization-doc>`.
       `(374) <https://github.com/attack68/rateslib/pull/374>`_
   * - Refactor
     - :red:`Minor Breaking Change!` *PPSpline* equality is now *True* if both spline
       coefficients are unsolved, i.e. *None*.
       `(374) <https://github.com/attack68/rateslib/pull/374>`_
   * - Refactor
     - The ``__repr__`` method of all *Curve* types, *FXRates* and *FXForwards* types, the *Solver*, *Schedule*,
       and all *Period*, *Leg* and *Instrument* types are changed for better display in associated
       packages.
       `(387) <https://github.com/attack68/rateslib/pull/387>`_
       `(388) <https://github.com/attack68/rateslib/pull/388>`_
       `(389) <https://github.com/attack68/rateslib/pull/389>`_
       `(390) <https://github.com/attack68/rateslib/pull/390>`_
       `(413) <https://github.com/attack68/rateslib/pull/413>`_
       `(416) <https://github.com/attack68/rateslib/pull/416>`_
       `(418) <https://github.com/attack68/rateslib/pull/418>`_
   * - Performance
     - Improve the speed of bond :meth:`~rateslib.instruments.FixedRateBond.ytm` calculations from about 750us to
       500us on average.
       `(380) <https://github.com/attack68/rateslib/pull/380>`_
   * - Bug
     - :class:`~rateslib.fx.FXRates` fix support for pickling which allows multithreading across CPU pools or
       external serialization.
       `(393) <https://github.com/attack68/rateslib/pull/393>`_
   * - Bug
     - The ``eom`` parameter for spec *"us_gb"* and *"us_gb_tsy"* and associated aliases is corrected to *True*.
       `(368) <https://github.com/attack68/rateslib/pull/368>`_
   * - Bug
     - Creating *IRS* or similar *Instruments* with a ``termination`` of "1b" or business days
       now correctly uses the specified calendar.
       `(378) <https://github.com/attack68/rateslib/pull/378>`_
   * - Bug
     - :class:`~rateslib.curves.ProxyCurve`, :class:`~rateslib.curves.CompositeCurve`, and
       :class:`~rateslib.curves.MultiCsaCurve` now correctly initialise a randomised curve ``id``
       when one is not provided.
       `(387) <https://github.com/attack68/rateslib/pull/387>`_
   * - Bug
     - Altered the *default specs* for ``eur_stir3`` to reflect a EURIBOR settlement, and
       ``aud_irs3`` to reflect a no-lagged publication.
       `(395) <https://github.com/attack68/rateslib/pull/395>`_
   * - Bug
     - The conventions for *"SE_GBB"* and *"SE_GB"* amended for
       T+2 settle instead of T+1, and the calculation for YTM adjusted for simple yield in the
       last coupon period.
       `(410) <https://github.com/attack68/rateslib/pull/410>`_
   * - Bug
     - IMM FRAs with an IMM roll date only need to define the IMM ``roll`` on leg1 and no longer
       also on leg2.
       `(409) <https://github.com/attack68/rateslib/pull/409>`_


1.4.0 (28th Aug 2024)
***********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Calendars
     - :meth:`~rateslib.calendars.add_tenor` acquires the new optional argument ``mod_days`` which, by
       default, negates the modification rule for day type tenors and applies it only to month and year type tenors.
   * - Calendars
     - Add :class:`~rateslib.calendars.NamedCal` for improved control of calendar serialization and loading.
   * - Instruments
     - Add a :meth:`~rateslib.instruments.FXOption.cashflows` method to generic :class:`~rateslib.instruments.FXOption`
       and also as a pre-requisite to :class:`~rateslib.periods.FXOptionPeriod`. This also allows the derivative
       method :meth:`~rateslib.instruments.Sensitivities.cashflows_table` to function for *FXOption*.
   * - Instruments
     - Add an internal routine to derive *FXOption* `expiry` and `delivery` according to FX market conventions using
       the new settlement calendar system introduced in v1.3.0.
   * - Instruments
     - Add ``eom`` parameter to *FXOptions* for exact expiry and delivery date calculation when given as string tenor.
   * - Instruments
     - The default ``calc_mode`` for *Bill*, *FixedRateBond*, *FloatRateNote* and *IndexFixedRateBond* is now
       separately configurable for each type.
   * - Instruments / Legs
     - Can now have *effective* and *termination* dates which are non-business dates
       in unmodified schedules.
   * - Surfaces
     - Add ``weights`` to :class:`~rateslib.fx_volatility.FXDeltaVolSurface` to give more control of temporal
       interpolation of volatility.
   * - Bug
     - Publicly exposed the :meth:`PPSpline.bsplmatrix <rateslib.splines.PPSplineF64.bsplmatrix>` function
       for displaying intermediate spline calculation results of the spline coefficient matrix.
   * - Bug
     - *Dual* and *Dual2* fix support for pickling which allows multithreading across CPU pools.
   * - Bug
     - Expose :meth:`~rateslib.dual.gradient` as a method in the *rateslib* public API.
   * - Bug
     - Expose :class:`~rateslib.calendars.NamedCal` as a class in the *rateslib* public API.
   * - Bug
     - :class:`~rateslib.instruments.IndexFixedRateBond` now correctly initialises when using a
       :class:`pandas.Series` as ``index_fixings`` argument.
   * - Bug
     - :class:`~rateslib.instruments.ZCIS` now raises if an ``index_base`` cannot be forecast from an *IndexCurve*
       and the value should be known and input directly, to avoid *Solver* calibration failures.
   * - Bug
     - ``npv`` and ``cashflows`` of a :class:`~rateslib.periods.FloatPeriod` now handle
       error messages regarding missing RFR fixings for an historical period which is only
       missing a single fixing.

1.3.0 (9th July 2024)
***********************

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Feature
     - Description
   * - Instruments
     - ``calc_mode`` of :class:`~rateslib.instruments.FixedRateBond` has been refactored to allow more standardised
       names. The existing modes are deprecated and will be removed in v2.0.
   * - Instruments
     - ``spec`` *"de_gb"*, *"fr_gb"*, *"it_gb"*, *"no_gb"* and *"nl_gb"*,
       added to :class:`~rateslib.instruments.FixedRateBond` to quickly create German, French,
       Italian, Norwegian and Dutch government bonds.
   * - Calendars
     - The `pandas` holiday and calendar system has been removed in favour of a rust implementation for
       calendar objects: :class:`~rateslib.calendars.Cal` and :class:`~rateslib.calendars.UnionCal`.
   * - Calendars
     - :red:`Breaking Change!` The :meth:`~rateslib.calendars.create_calendar` methods is deprecated and
       modified to accept different input arguments.
   * - Calendars
     - Calendar string parsing has been enhanced to allow associated settlement calendars, and
       automatic creation of a :class:`~rateslib.calendars.UnionCal` object. E.g. *"tgt,ldn|nyc"*.
   * - Calendars
     - The Tokyo calendar *'tyo'* has been added to align with TONA publication. The FED calendar *'fed'* has also been
       added. The Sydney calendar *"syd"* has been added to align with AONIA publication.
   * - Calendars
     - JSON serialisation/deserialisation of :class:`~rateslib.calendars.Cal`
       and :class:`~rateslib.calendars.UnionCal` added for saving/loading from database or file.
   * - Calendars
     - The new DCF method *'Bus252'* is added to allow Brazilian type calculations.
   * - Dual
     - JSON serialisation/deserialisation of :class:`~rateslib.dual.Dual`
       and :class:`~rateslib.dual.Dual2` added for saving/loading from database or file.
   * - FXRates
     - The :class:`~rateslib.fx.FXRates` class has been delegated to the Rust extension to improve performance.
   * - Performance
     - Algorithm for :class:`~rateslib.fx.FXRates` generation is modified to improve the speed of instance
       construction for a larger number of currencies.
   * - FX Volatility
     - :meth:`~rateslib.fx_volatility.FXDeltaVolSmile.get_from_strike` on both *Smiles* and *Surfaces* has
       been refactored to remove the unnecessary ``phi`` argument.
   * - Bug
     - :class:`~rateslib.instruments.ZCS` now raises if fixed frequency is given as "Z".
   * - Bug
     - :meth:`~rateslib.instruments.FixedRateBond.rate` method of a *FixedRateBond* now correctly
       returns the local currency price or yield-to-maturity without being wrongly converted by a
       ``base`` FX rate, if an FX object is also supplied to the pricing formula.
   * - Bug
     - :class:`~rateslib.instruments.FXOption` initialised with ``metric`` no longer
       raises if an alternate dynamic ``metric`` is requested as override in the
       :meth:`~rateslib.instruments.FXOption.rate` method.
   * - Bug
     - Setting and resetting some types of values (namely by-reference stored values) of the ``defaults`` object
       is no longer ineffective.
   * - Bug
     - Solving acyclic *FXForwards* systems is now stable for all orderings of currencies, and does not depend
       on a well chosen ``base`` currency.
   * - Bug
     - Converting an `fx_array` associated with the :class:`~rateslib.fx.FXRates` into second order for AD
       calculations now captures second order FX derivatives correctly by rebuilding the array, instead of a
       direct conversion setting second order derivatives to zero.
   * - Bug
     - Entering the *"single_vol"* ``metric`` into the :meth:`~rateslib.instruments.FXBrokerFly.rate` method
       of a :class:`~rateslib.instruments.FXBrokerFly` no longer raises.
   * - Errors
     - Improved messages when missing `fx` objects for pricing :class:`~rateslib.instruments.FXExchange`.


1.2.2 (31st May 2024)
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
     - New methods :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual`,
       :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual2`,
       :meth:`~rateslib.splines.PPSplineF64.ppdnev_single_dual`,
       and :meth:`~rateslib.splines.PPSplineF64.ppdnev_single_dual2` have been added to
       ensure correct handling of AD with regards to both x-axis and y-axis variables. See
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
     - New pricing components :class:`~rateslib.fx_volatility.FXDeltaVolSmile` and
       :class:`~rateslib.fx_volatility.FXDeltaVolSurface`
       have been added
       to allow pricing of single expiry *FX Options* with a *Smile* interpolated over a *Delta*
       axis. See :ref:`FX volatility construction <c-fx-smile-doc>`.
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

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/thumb_coding_3.png
         :alt: Coding Interest Rates: FX, Swaps and Bonds
         :target: https://www.amazon.com/dp/0995455554
         :width: 145
         :align: center

   .. container:: rightside60

      The publication to the left, *"Coding Interest Rates: FX, Swaps and Bonds"*
      documents the API architecture and mathematical algorithms for its objects
      upto and including the version one release of *rateslib*.

.. raw:: html

   <div class="clear" style="text-align: center; padding: 1em 0em 1em;"></div>

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
