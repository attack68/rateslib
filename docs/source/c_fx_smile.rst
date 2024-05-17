.. _c-fx-smile-doc:

.. ipython:: python
   :suppress:

   import warnings
   warnings.filterwarnings('always')
   from rateslib.solver import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

********************
FX Volatility Smile
********************

.. warning::

   FX volatility products in *rateslib* are not in stable status. Their API and/or object
   interactions may incur breaking changes in upcoming releases as it matures and other
   classes, such as a *VolSurface* are added.

The ``rateslib.fx_volatility`` module includes an :class:`~rateslib.fx_volatility.FXDeltaVolSmile`
class which can be used to price *FX Options* and *FX Option Strategies*.

.. autosummary::
   rateslib.fx_volatility.FXDeltaVolSmile

Introduction
************

The *FXDeltaVolSmile* is parametrised by a series of *(delta, vol)* node points
interpolated by a cubic spline. This interpolation is automatically constructed with knot
sequences that adjust to the number of given ``nodes``. One node will create a constant
vol level, and two nodes will create a straight line gradient. More nodes (appropriately calibrated)
will create a traditional smile shape.

An *FXDeltaVolSmile* must also be initialised with an ``eval_date`` which serves the same
purpose as the initial node point on a *Curve*, and indicates *'today'*. There must also be an ``expiry``, and
options priced with this *Smile* must have an equivalent expiry or errors will be raised.
Finally, the ``delta_type`` of the *Smile* must be specified so that its delta index is well
defined.

Constructing a Smile
*********************

The following data describes *Instruments* to calibrate the EURUSD FX volatility surface on 7th May 2024.
We will take a cross-section of this data, at the 3-week expiry (28th May 2024), and create an *FXDeltaVolSmile*.

.. image:: _static/fx_eurusd_3m_surf.PNG
  :alt: EURUSD FX volatility surface prices on 7th May 2024
  :width: 489

Since EURUSD is **not** premium adjusted and the premium currency is USD we will match the *Smile* with this
definition and set it to a ``delta_type`` of *'spot'*, matching the market convention of these quoted instruments.
Since we have 5 calibrating instruments we require 5 degrees of freedom.

.. ipython:: python

   smile = FXDeltaVolSmile(
       nodes={
           0.10: 10.0,
           0.25: 10.0,
           0.50: 10.0,
           0.75: 10.0,
           0.90: 10.0,
       },
       eval_date=dt(2024, 5, 7),
       expiry=dt(2024, 5, 28),
       delta_type="spot",
       id="eurusd_3w_smile"
   )

The above *Smile* is initialised as a flat vol at 10%. In order to calibrate it we need to create the pricing
instruments, given in the market prices data table.

Since we are using multi-currency derivatives we will also need to setup an :class:`~rateslib.fx.FXForwards`
framework. We will do this simultaneously using other prevailing market data.

.. ipython:: python

   # Define the interest rate curves for EUR, USD and X-Ccy basis
   eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
   eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
   usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
   # Create an FX Forward market with spot FX rate data
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
       fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
   )
   # Setup the Solver instrument calibration for rates Curves and vol Smiles
   option_args=dict(
       pair="eurusd", expiry=dt(2024, 5, 28), calendar="tgt", delta_type="spot",
       curves=[None, "eurusd", None, "usdusd"], vol="eurusd_3w_smile"
   )
   solver = Solver(
       curves=[eureur, eurusd, usdusd, smile],
       instruments=[
           IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
           IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
           FXSwap(dt(2024, 5, 9), "3W", currency="eur", leg2_currency="usd", curves=[None, "eurusd", None, "usdusd"]),
           FXStraddle(strike="atm_delta", **option_args),
           FXRiskReversal(strike=["-25d", "25d"], **option_args),
           FXRiskReversal(strike=["-10d", "10d"], **option_args),
           FXBrokerFly(strike=["-25d", "atm_delta", "25d"], **option_args),
           FXBrokerFly(strike=["-10d", "atm_delta", "10d"], **option_args),
       ],
       s=[3.90, 5.32, 8.85, 5.493, -0.157, -0.289, 0.071, 0.238],
       fx=fxf,
   )
   smile.plot()

.. container:: twocol

   .. container:: leftside50

      .. plot::
         :caption: Rateslib Vol Smile

         from rateslib.curves import Curve
         from rateslib.instruments import *
         from rateslib.fx_volatility import FXDeltaVolSmile
         from rateslib.fx import FXRates, FXForwards
         from rateslib.solver import Solver
         import matplotlib.pyplot as plt
         from datetime import datetime as dt
         smile = FXDeltaVolSmile(
             nodes={
                 0.10: 10.0,
                 0.25: 10.0,
                 0.50: 10.0,
                 0.75: 10.0,
                 0.90: 10.0,
             },
             eval_date=dt(2024, 5, 7),
             expiry=dt(2024, 5, 28),
             delta_type="spot",
             id="eurusd_3w_smile"
         )
         # Define the interest rate curves for EUR, USD and X-Ccy basis
         eureur = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="tgt", id="eureur")
         eurusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, id="eurusd")
         usdusd = Curve({dt(2024, 5, 7): 1.0, dt(2024, 5, 30): 1.0}, calendar="nyc", id="usdusd")
         # Create an FX Forward market with spot FX rate data
         fxf = FXForwards(
             fx_rates=FXRates({"eurusd": 1.0760}, settlement=dt(2024, 5, 9)),
             fx_curves={"eureur": eureur, "usdusd": usdusd, "eurusd": eurusd},
         )
         # Setup the Solver instrument calibration for rates Curves and vol Smiles
         option_args=dict(
             pair="eurusd", expiry=dt(2024, 5, 28), calendar="tgt", delta_type="spot",
             curves=[None, "eurusd", None, "usdusd"], vol="eurusd_3w_smile"
         )
         solver = Solver(
             curves=[eureur, eurusd, usdusd, smile],
             instruments=[
                 IRS(dt(2024, 5, 9), "3W", spec="eur_irs", curves="eureur"),
                 IRS(dt(2024, 5, 9), "3W", spec="usd_irs", curves="usdusd"),
                 FXSwap(dt(2024, 5, 9), "3W", currency="eur", leg2_currency="usd", curves=[None, "eurusd", None, "usdusd"]),
                 FXStraddle(strike="atm_delta", **option_args),
                 FXRiskReversal(strike=["-25d", "25d"], **option_args),
                 FXRiskReversal(strike=["-10d", "10d"], **option_args),
                 FXBrokerFly(strike=["-25d", "atm_delta", "25d"], **option_args),
                 FXBrokerFly(strike=["-10d", "atm_delta", "10d"], **option_args),
             ],
             s=[3.90, 5.32, 8.85, 5.493, -0.157, -0.289, 0.071, 0.238],
             fx=fxf,
         )
         fig, ax, line = smile.plot()
         plt.show()
         plt.close()

   .. container:: rightside50

      |
      |

      .. figure:: _static/fx_eurusd_3w_smile.PNG
         :alt: BBG FENICS EURUSD Smile on 7th May 2024
         :width: 320

         BBG Fenics Vol Smile
