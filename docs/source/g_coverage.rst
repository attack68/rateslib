.. _coverage-doc:

***********
Coverage
***********

The **test coverage** of this library is extensive. There are over 900 unit tests
with the following coverage report:

.. list-table:: Test Coverage Report: as of commit #b5a19547
   :widths: 52 16 16 16
   :header-rows: 1

   * - Module
     - Lines
     - Miss
     - Cover
   * - __init__.py
     - 35
     - 1
     - 97%
   * - calendars.py
     - 166
     - 0
     - 100%
   * - curves.py
     - 314
     - 1
     - 99%
   * - defaults.py
     - 48
     - 4
     - 92%
   * - dual.py
     - 321
     - 0
     - 100%
   * - fx.py
     - 465
     - 43
     - 91%
   * - instruments.py
     - 1024
     - 24
     - 98%
   * - legs.py
     - 333
     - 3
     - 99%
   * - periods.py
     - 357
     - 0
     - 100%
   * - scheduling.py
     - 332
     - 0
     - 100%
   * - solver.py
     - 563
     - 12
     - 98%
   * - splines.py
     - 108
     - 0
     - 100%
   * - **TOTAL**
     - **7444**
     - **88**
     - **99%**

It should be noted that test coverage is not the same as hypothesis testing and ensuring
that the results of combinations of input arguments has the desired effect. This is
considered and attempted in various ways but the amount of possible combinations,
for example when combining schedules, legs and derivatives is quite exhaustive and
the disclaimer is that some edge cases may well be missed.

*****
Style
*****

*Rateslib* uses `black` as a PEP8 compliant opinionated code styler.

******
Typing
******

*Rateslib* strives to ultimately be fully explicitly typed, but this
has not yet been implemented, and will form part of its phase 3 development.