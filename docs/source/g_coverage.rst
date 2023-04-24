.. _coverage-doc:

***********
Coverage
***********

The **test coverage** of this library is extensive. There are over 900 unit tests
with the following coverage report:

.. list-table:: Test Coverage Report: as of commit #f0cdcc33
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
     - 286
     - 0
     - 100%
   * - defaults.py
     - 47
     - 4
     - 91%
   * - dual.py
     - 305
     - 0
     - 100%
   * - fx.py
     - 465
     - 43
     - 91%
   * - instruments.py
     - 837
     - 49
     - 94%
   * - legs.py
     - 333
     - 35
     - 89%
   * - periods.py
     - 286
     - 2
     - 99%
   * - scheduling.py
     - 332
     - 0
     - 100%
   * - splines.py
     - 108
     - 0
     - 100%
   * - **TOTAL**
     - **6727**
     - **155**
     - **98%**

It should be noted that test coverage is not the same as hypothesis testing and ensuring
that the results of combinations of input arguments has the desired effect. This is
considered and attempted in various ways but the amount of possible combinations,
for example when combining schedules, legs and derivatives is quite exhaustive and
the disclaimer is that some edge cases may well be missed.