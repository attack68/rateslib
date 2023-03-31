.. _coverage-doc:

***********
Coverage
***********

The **test coverage** of this library is extensive. There are over 450 tests with the
following coverage report:

.. list-table:: Test Coverage Report
   :widths: 52 16 16 16
   :header-rows: 1

   * - Module
     - Lines
     - Miss
     - Cover
   * - dual.py
     - 272
     - 0
     - 100%
   * - interpolation.py
     - 24
     - 4
     - 83%
   * - scheduling.py
     - 478
     - 58
     - 88%
   * - splines.py
     - 88
     - 2
     - 98%

It should be noted that test coverage is not the same as hypothesis testing and ensuring
that the results of combinations of input arguments has the desired effect. This is
considered and attempted in various ways but the amount of possible combinations,
for example when combining schedules, legs and derivatives is quite exhaustive and
the disclaimer is that some edge cases may well be missed.