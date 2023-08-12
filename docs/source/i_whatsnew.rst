.. _whatsnew-doc:

**************
Release Notes
**************

More detailed release notes started being recorded after version 0.3.x

0.4.0 (12 Aug 2023)
********************

Refactors and Enhancements
--------------------------

- Added ``split_notional`` to ``FXSwap`` to more accurately reflect the interbank traded product.
- Added an ``FXExchange`` class to provide booking FXSpot or FXForward trades of simple FX
  exchanges.
- Remove all *LegExchange* types and add ``initial_exchange`` and
  ``final_exchange`` as arguments to basic *Legs* to replace the functionality.
- Added historic fixing data until end July for ESTR, SOFR,
  SWESTR, SONIA and NOWA, for testing and validation.
- Caching calendars improves general performance by about 10%.
- Collateral tags were added to *Curves* to permit the new method ``cashflows_table`` which
  tabulates futures cashflows according to currency and collateral type.

Bug Fixes
---------

- When performing operations on *CompositeCurves* the resultant curve now inherits
  the ``multi_csa`` parameters.
- Float Fixing exposure tables were marginally overestimated by ignoring
  discounting effects.
- NumPy.float128 datatype is not available on Windows and caused loading errors.
- Fixed and enhanced the holiday calendars: 'ldn', 'tgt', 'nyc', 'stk', 'osl',
  and 'zur', and validated historic fixings against the historic fixing data.
- Now allow *CompositeCurve* to be constructed from *ProxyCurve* and *Curve*
  combinations.
- The ``payment_lag_exchange`` parameter for ``FXSwap`` was removed in favour of using
  ``payment_lag``.
