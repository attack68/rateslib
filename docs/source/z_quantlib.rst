.. _cook-quantlib-doc:

Comparing Curve Building and Instrument Pricing with QuantLib
****************************************************************
This document offers a brief comparison between *rateslib* and *QuantLib* for constructing a *Curve*. In this example we build a *Curve* whose index is STIBOR-3M. 
The evaluation includes *Curve* creation and pricing an :class:`~rateslib.instruments.IRS` using both libraries. Let's start with *QuantLib*:

.. code-block:: python

   import QuantLib as ql

   # Set the evaluation date
   ql.Settings.instance().evaluationDate = ql.Date(3,1,2023) 

   # Define market data for the curve
   data = {
       '1Y': 3.45,
       '2Y': 3.4,
       '3Y': 3.37,
       '4Y': 3.33,
       '5Y': 3.2775,
       '6Y': 3.235,
       '7Y': 3.205,
       '8Y': 3.1775,
       '9Y': 3.1525,
       '10Y': 3.1325,
       '12Y': 3.095,
       '15Y': 3.0275,
       '20Y': 2.92,
       '25Y': 2.815,
       '30Y': 2.6925
   } 


The next step is to define the *IRSs* for bootstrapping using `ql.SwapRateHelper`:

.. code-block:: python


   # Create a custom ibor index for the floating leg
   ibor_index = ql.IborIndex(
       'STIBOR3M',           # Name of the index
       ql.Period('3M'),      # Maturity
       0,                    # Fixing days
       ql.SEKCurrency(),     # Currency
       ql.Sweden(),          # Calendar
       ql.ModifiedFollowing, # Convention
       False,                # EOM convention
       ql.Actual360()        # Daycount
   ) 
   # Create the bootstrapping instruments using helpers
   swap_helpers = [ql.SwapRateHelper(
      ql.QuoteHandle(ql.SimpleQuote(rate/100.0)),   # Quote
      ql.Period(term),                              # Maturity
      ql.Sweden(),                                  # Calendar
      ql.Annual,                                    # Fixed payments
      ql.ModifiedFollowing,                         # Convention
      ql.Actual360(),                               # Daycount
   ibor_index) for term, rate in data.items()]

Finally, the curve is built using `ql.PiecewiseLogLinearDiscount`:

.. code-block:: python
   
   curve = ql.PiecewiseLogLinearDiscount(0, ql.Sweden(), swap_helpers, ql.Actual360())

The *rateslib* code below will replicate the *Curve* creation, but note the difference in handling the nodes (pillar dates) of the *Curve*:

.. ipython:: python

   import rateslib as rl

   # Define market data for the curve
   data = {
       '1Y': 3.45,
       '2Y': 3.4,
       '3Y': 3.37,
       '4Y': 3.33,
       '5Y': 3.2775,
       '6Y': 3.235,
       '7Y': 3.205,
       '8Y': 3.1775,
       '9Y': 3.1525,
       '10Y': 3.1325,
       '12Y': 3.095,
       '15Y': 3.0275,
       '20Y': 2.92,
       '25Y': 2.815,
       '30Y': 2.6925
   } 

   curve = rl.Curve(
      id="curve",                    # Curve ID
      convention = 'act360',         # Daycount
      calendar = 'stk',              # Swedish Calendar 
      modifier = 'MF',               # Modified Following
      interpolation = 'log_linear',  # Interpolation Method 
      nodes={
         **{rl.dt(2023, 1, 3): 1.0}, # Initial node always starts at 1.0
         **{rl.add_tenor(rl.dt(2023, 1, 3), tenor, "MF", "stk"): 1.0 for tenor in data.keys()}
         },
   )

.. warning::
   Note that *rateslib* will determine the discount factors (DFs) based at the provided input node dates. *QuantLib*, which uses bootstrapping, sets these dates based on the maturity dates of the *Instruments* by default to ensure a sound bootstrapping routine.
   Thus to replicate the result from *QuantLib*, the function :meth:`add_tenor()<rateslib.calendars.add_tenor>` is used to find
   the adjusted maturity dates for each *Instrument* and use those values as input to our *Curve*.
   
The next step is to create the *Instruments* and call the :class:`~rateslib.solver.Solver`:

.. ipython:: python

   # Create the instrument attributes for the solver corresponding to our helpers in QuantLib
   instr_args= dict(
      effective=rl.dt(2023, 1, 3),  
      frequency="A",                 
      calendar="stk",                
      convention="act360",           
      currency="sek",
      curves="curve",
      payment_lag=0,
   )

   # Solve for the discount factors
   solver = rl.Solver(
      curves=[curve],
      instruments=[rl.IRS(termination=_, **instr_args) for _ in data.keys()],
      s=[_ for _ in data.values()]
   )
   curve.nodes

Finally the result beween the two libraries is summarized in the table below: 

.. table:: Discount Factors from rateslib and QuantLib

  ============= ============= ============= ============== 
    Curve Nodes    RatesLib      QuantLib       Residual    
  ============= ============= ============= ============== 
    2023-01-03         1             1             0        
    2024-01-03    0.966203023   0.966203023   -1.34004E-13  
    2025-01-03    0.93439395    0.93439395    -4.45088E-13  
    2026-01-05    0.903918458   0.903918458   -1.22391E-12  
    2027-01-04    0.875578174   0.875578174   -2.18003E-12  
    2028-01-03    0.849391648   0.849391648    -3.467E-12   
    2029-01-03    0.824236176   0.824236176   -5.21694E-12  
    2030-01-03    0.799874114   0.799874114   -7.35501E-12  
    2031-01-03    0.776572941   0.776572941    -1.003E-11   
    2032-01-05    0.754095707   0.754095707   -1.34019E-11  
    2033-01-03    0.732456627   0.732456627   -2.72941E-11  
    2035-01-03    0.691621953   0.691621953   -7.6532E-11   
    2038-01-04    0.637877344   0.637877345   -2.43384E-10  
    2043-01-05    0.562978818   0.562978819   -5.81426E-10  
    2048-01-03    0.503558382   0.503558381   1.48542E-10   
    2053-01-03    0.459970336   0.45997033     5.2457E-09   
  ============= ============= ============= ============== 


Given that the term structure that have been created by both libraries, the next step is to value an *IRS*. Starting with *QuantLib*:

.. code-block:: python

  # Link the zero rate curve to be used as forward and discounting
  yts = ql.RelinkableYieldTermStructureHandle()
  yts.linkTo(curve)
  engine = ql.DiscountingSwapEngine(yts)

  # Define the maturity of our swap
  maturity = ql.Period("2y")
  # Create a custom Ibor index for the floating leg
  custom_ibor_index = ql.IborIndex(
      "Ibor",
      ql.Period("1Y"),
      0,
      ql.SEKCurrency(),
      ql.Sweden(),
      ql.ModifiedFollowing,
      False,
      ql.Actual360(),
      yts,
  )
  fixed_rate = 0.03269
  forward_start = ql.Period("0D")
  # Create the swap using the helper class MakeVanillaSwap
  swap = ql.MakeVanillaSwap(
      maturity,
      custom_ibor_index,
      fixed_rate,
      forward_start,
      Nominal=10e7,
      pricingEngine=engine,
      fixedLegDayCount=ql.Actual360(),
  )

Above we have specified the attributes of our *IRS* in *QuantLib* and now we want to price it and extract the NPVs and the corresponding cashflows:

.. code-block:: python

  import pandas as pd

  fixed_cashflows = pd.DataFrame(
      [
          {
              "Type": "FixedPeriod",
              "accrualStart": cf.accrualStartDate().ISO(),
              "accrualEnd": cf.accrualEndDate().ISO(),
              "paymentDate": cf.date().ISO(),
              "df": curve.discount(cf.accrualEndDate()),
              "rate": cf.rate(),
              "cashflow": cf.amount(),
              "npv": -curve.discount(cf.accrualEndDate()) * cf.amount(),
          }
          for cf in map(ql.as_fixed_rate_coupon, swap.leg(0))
      ]
  )

  float_cashflows = pd.DataFrame(
      [
          {
              "Type": "FloatPeriod",
              "accrualStart": cf.accrualStartDate().ISO(),
              "accrualEnd": cf.accrualEndDate().ISO(),
              "paymentDate": cf.date().ISO(),
              "df": curve.discount(cf.accrualEndDate()),
              "rate": cf.rate(),
              "cashflow": cf.amount(),
              "npv": curve.discount(cf.accrualEndDate()) * cf.amount(),
          }
          for cf in map(ql.as_floating_rate_coupon, swap.leg(1))
      ]
  )

  ql_cashflows = pd.concat([fixed_cashflows, float_cashflows])

This results in the following cashflows:

.. table:: Cashflows attributes from QuantLib

  +-------+----------+----------+-----------+---------+------------+-------------+
  | Type  | accStart | accEnd   |df         | rate    | cashflow   | npv         |
  +=======+==========+==========+===========+=========+============+=============+
  | Fixed | 23-01-03 | 24-01-03 | 0.9662030 | 0.03269 | 3314402.77 | -3202385.98 |
  | Fixed | 24-01-03 | 25-01-03 | 0.9343939 | 0.03269 | 3323483.33 | -3105442.72 |
  | Float | 23-01-03 | 24-01-03 | 0.9662030 | 0.03450 | 3497916.66 | 3379697.65  |
  | Float | 24-01-03 | 25-01-03 | 0.9343939 | 0.03348 | 3404246.45 | 3180907.29  |
  +-------+----------+----------+-----------+---------+------------+-------------+

Which compared with *rateslib*


.. ipython:: python

  irs = rl.IRS(
      effective=rl.dt(2023, 1, 3),
      termination="2Y",
      frequency="A",
      calendar="stk",
      currency="sek",
      fixed_rate=3.269,
      convention="Act360",
      notional=100e6,
      curves=["curve"],
      payment_lag=0,
      modifier='F'
  )

  rl_cashflows = irs.cashflows(curves=[curve])

Results in the following table:

.. table:: Cashflows attributes from rateslib

  +-------+----------+----------+-----------+--------+-------------+-------------+
  | Type  | AccStart | Acc End  | DF        | Rate   | Cashflow    | NPV         |
  +=======+==========+==========+===========+========+=============+=============+
  | Fixed | 23-01-03 | 24-01-03 | 0.9662030 | 3.2690 | -3314402.77 | -3202385.98 |
  | Fixed | 24-01-03 | 25-01-03 | 0.9343939 | 3.2690 | -3323483.33 | -3105442.72 |
  | Float | 23-01-03 | 24-01-03 | 0.9662030 | 3.4500 | 3497916.66  | 3379697.65  |
  | Float | 24-01-03 | 25-01-03 | 0.9343939 | 3.3484 | 3404246.45  | 3180907.29  |
  +-------+----------+----------+-----------+--------+-------------+-------------+

Which is identical to the *QuantLib* result. 
If you're interested in delving deeper into the calculation of DFs by *rateslib* and *QuantLib*, you may find some insights in this `blog post <https://xiar-fatah.github.io/2023/11/14/rateslib-bootstrapping.html>`_.