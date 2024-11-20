.. _coverage-doc:

***********
Coverage
***********

The **test coverage** of this library is extensive. There are over 900 unit tests
with the following coverage report:

**Test Coverage Report: as of commit #4b848b6b**

.. code-block::

   Name                                            LINES   MISS      %
   -------------------------------------------------------------------
   \__init__.py                                       38      0   100%
   \_spec_loader.py                                   61      2    97%
   \calendars\__init__.py                             98      0   100%
   \calendars\dcfs.py                                140      8    94%
   \calendars\rs.py                                   52      2    96%
   \curves\__init__.py                                 2      0   100%
   \curves\curves.py                                 674     15    98%
   \curves\rs.py                                      73      8    89%
   \default.py                                       118     10    92%
   \dual\__init__.py                                 105      8    92%
   \dual\variable.py                                 125     16    87%
   \fx\__init__.py                                     3      0   100%
   \fx\fx_forwards.py                                339     20    94%
   \fx\fx_rates.py                                   139      0   100%
   \fx_volatility.py                                 332     36    89%
   \instruments\__init__.py                            8      0   100%
   \instruments\bonds\__init__.py                      5      0   100%
   \instruments\bonds\conventions\__init__.py         51      4    92%
   \instruments\bonds\conventions\accrued.py          49      1    98%
   \instruments\bonds\conventions\discounting.py      54      3    94%
   \instruments\bonds\futures.py                     185     26    86%
   \instruments\bonds\securities.py                  474     27    94%
   \instruments\core.py                              305     12    96%
   \instruments\fx_volatility.py                     353     14    96%
   \instruments\generics.py                          162     11    93%
   \instruments\rates_derivatives.py                 355      6    98%
   \instruments\rates_multi_ccy.py                   326      1    99%
   \json.py                                           10      0   100%
   \legs.py                                          560      0   100%
   \periods.py                                      1176     67    94%
   \scheduling.py                                    362      2    99%
   \solver.py                                        704     19    97%
   \splines.py                                        14      0   100%
   -------------------------------------------------------------------
   TOTAL                                            7452    318    96%

It should be noted that test coverage is not the same as hypothesis testing and ensuring
that the results of combinations of input arguments has the desired effect. This is
considered and attempted in various ways but the amount of possible combinations,
for example when combining schedules, legs and derivatives is quite exhaustive and
the disclaimer is that some edge cases may well be missed.

*****
Style
*****

*Rateslib* uses `black` and `ruff` as a PEP8 compliant opinionated code styler.

******
Typing
******

*Rateslib* strives to ultimately be fully explicitly typed, but this
has not yet been implemented, and will form part of its phase 3 development.