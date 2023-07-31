.. _whatsnew-doc:

**************
Release Notes
**************

Proper release notes started being recorded after version 0.3.0

0.4.0
******

- Bug: allow *CompositeCurve* to be constructed from *ProxyCurve* and *Curve*
  combinations.
- Refactor: remove *LegExchange* types and add ``initial_exchange`` and
  ``final_exchange`` as arguments to basic *Legs* to replace the functionality.