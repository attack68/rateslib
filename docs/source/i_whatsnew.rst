.. _whatsnew-doc:

**************
Release Notes
**************

Proper release notes started being recorded after version 0.3.x

0.4.0 (not released)
********************

- Bug: fixed and enhanced the holiday calendars: 'ldn', 'tgt', 'nyc', 'stk', 'osl',
  and 'zur', and validated historic fixings against the historic fixing data.
- Enhancement: added historic fixing data until end July for ESTR, SOFR,
  SWESTR, SONIA and NOWA, for testing and validation.
- Bug: allow *CompositeCurve* to be constructed from *ProxyCurve* and *Curve*
  combinations.
- Refactor: remove *LegExchange* types and add ``initial_exchange`` and
  ``final_exchange`` as arguments to basic *Legs* to replace the functionality.