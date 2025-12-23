.. _coverage-doc:

***********
Coverage
***********

The **test coverage** of this library is extensive. There are over 2900 unit tests
with the following coverage report:

**Test Coverage Report: as of commit #89ea2154**

.. code-block::

   Name                                                           Stmts   Miss  Cover   Missing
   --------------------------------------------------------------------------------------------
   python/rateslib/__init__.py                                       38      0   100%
   python/rateslib/_spec_loader.py                                   63     54    14%   17-99
   python/rateslib/calendars/__init__.py                            110      2    98%   18, 346
   python/rateslib/calendars/dcfs.py                                143     10    93%   13, 41-42, 128, 196, 198, 200, 223, 242, 275
   python/rateslib/calendars/rs.py                                   73     14    81%   10, 161-170, 184, 188, 195, 199
   python/rateslib/curves/__init__.py                                 2      0   100%
   python/rateslib/curves/_parsers.py                               107      4    96%   12, 201, 216, 247
   python/rateslib/curves/curves.py                                 760     29    96%   44, 235, 447, 475, 1132, 1218, 1221, 1229-1230, 1472, 1475, 1509, 1567, 1770, 2377, 2457, 2504, 2563, 2644, 2646, 2657, 2676, 2680, 2684, 2688, 2692, 2696, 2818, 2923
   python/rateslib/curves/rs.py                                      76      9    88%   27, 62, 74, 85-87, 97, 101, 120
   python/rateslib/default.py                                       121     10    92%   294-381, 410-420
   python/rateslib/dual/__init__.py                                   9      0   100%
   python/rateslib/dual/newton.py                                    95      1    99%   13
   python/rateslib/dual/quadratic.py                                 19      6    68%   8, 54-57, 71, 91
   python/rateslib/dual/utils.py                                    134     19    86%   15, 34, 43, 53, 117-120, 124-127, 162, 171, 179, 186, 323, 327, 345-348
   python/rateslib/dual/variable.py                                 131     18    86%   14, 69, 88, 91, 94, 97, 100, 116, 133, 156, 173, 189, 212-217
   python/rateslib/fx/__init__.py                                     3      0   100%
   python/rateslib/fx/fx_forwards.py                                367      9    98%   20, 814, 816, 830, 1027, 1036, 1045, 1135, 1137
   python/rateslib/fx/fx_rates.py                                   150      1    99%   28
   python/rateslib/fx_volatility.py                                 364     38    90%   391-402, 537-567, 733, 735, 737, 1140-1144, 1384
   python/rateslib/instruments/__init__.py                            9      0   100%
   python/rateslib/instruments/base.py                              119      1    99%   21
   python/rateslib/instruments/bonds/__init__.py                      5      0   100%
   python/rateslib/instruments/bonds/conventions/__init__.py         55      5    91%   10, 201, 358-360
   python/rateslib/instruments/bonds/conventions/accrued.py          53      2    96%   11, 98
   python/rateslib/instruments/bonds/conventions/discounting.py      58      4    93%   10, 145, 227-228
   python/rateslib/instruments/bonds/futures.py                     180     28    84%   19, 295, 451-508, 537, 589, 842-844, 947, 955
   python/rateslib/instruments/bonds/securities.py                  567     59    90%   47, 276, 288-289, 427-433, 758, 762-768, 813, 852, 854, 860, 862-863, 925-951, 1356, 1747, 1755, 1874, 2363-2366, 2418-2420, 2645, 2655, 2676, 2737, 3001, 3009, 3138-3139
   python/rateslib/instruments/credit/__init__.py                     2      0   100%
   python/rateslib/instruments/credit/derivatives.py                 45      2    96%   18, 112
   python/rateslib/instruments/fx_volatility/__init__.py              3      0   100%
   python/rateslib/instruments/fx_volatility/strategies.py          283     16    94%   22, 73-79, 87-93, 119, 354, 1252-1254
   python/rateslib/instruments/fx_volatility/vanilla.py             151      5    97%   27-29, 267, 378-379
   python/rateslib/instruments/generics.py                          180     12    93%   27, 137, 264-266, 269, 272, 275, 423, 431, 643, 665-671
   python/rateslib/instruments/rates/__init__.py                      4      0   100%
   python/rateslib/instruments/rates/inflation.py                   100      2    98%   17, 559
   python/rateslib/instruments/rates/multi_currency.py              371      2    99%   43, 268
   python/rateslib/instruments/rates/single_currency.py             288     13    95%   33-35, 678, 703, 1565, 1567, 1569, 1571, 1692, 1701, 1744, 1822, 1848
   python/rateslib/instruments/sensitivities.py                      57      3    95%   17, 144, 155
   python/rateslib/instruments/utils.py                             132     23    83%   15, 136, 162, 169-182, 241-242, 283-284, 288, 302-304, 308-310
   python/rateslib/json.py                                           11      0   100%
   python/rateslib/legs.py                                          669      8    99%   35, 1374, 1491, 1567, 1913, 2413, 2692, 3054
   python/rateslib/mutability/__init__.py                            70      2    97%   12, 119
   python/rateslib/periods/__init__.py                                7      0   100%
   python/rateslib/periods/base.py                                   52      1    98%   14
   python/rateslib/periods/cashflow.py                               97      2    98%   18, 360
   python/rateslib/periods/credit.py                                122      6    95%   19, 198-200, 308, 419
   python/rateslib/periods/fx_volatility.py                         463     60    87%   32, 155, 158, 220, 234, 303, 342, 458, 475, 970, 1143-1225, 1250, 1271
   python/rateslib/periods/index.py                                 143      7    95%   21, 112-116, 143, 172, 205, 324
   python/rateslib/periods/rates.py                                 445     14    97%   30, 144, 756, 763, 1197, 1339, 1418-1424, 1426-1432, 1569, 1662
   python/rateslib/periods/utils.py                                 116     11    91%   15, 196-197, 205, 231-234, 239, 250, 277, 282
   python/rateslib/scheduling.py                                    356      4    99%   27, 946, 1111, 1606
   python/rateslib/solver.py                                        729     23    97%   36-40, 1125, 1255, 1324, 1335, 1430, 1482, 1563, 1732, 1742, 1750, 2036, 2065-2072, 2221, 2230-2233, 2257
   python/rateslib/splines.py                                        25      1    96%   11
   --------------------------------------------------------------------------------------------
   TOTAL                                                           8732    540    94%


It should be noted that test coverage is not the same as hypothesis testing and ensuring
that the results of combinations of input arguments has the desired effect. This is
considered and attempted in various ways but the amount of possible combinations,
for example when combining schedules, legs and derivatives is quite exhaustive and
the disclaimer is that some edge cases may well be missed.

*****
Style
*****

*Rateslib* uses `ruff` as a PEP8 compliant opinionated code styler.

******
Typing
******

*Rateslib* is statically typed and uses `mypy` as a type checker.