.. _whatsnew-doc:

**************
Release Notes
**************

Proper release notes started being recorded after version 0.3.x

0.4.0 (not released)
********************

Refactors and Enhancements
--------------------------

- Remove all *LegExchange* types and add ``initial_exchange`` and
  ``final_exchange`` as arguments to basic *Legs* to replace the functionality.
- Added historic fixing data until end July for ESTR, SOFR,
  SWESTR, SONIA and NOWA, for testing and validation.

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
