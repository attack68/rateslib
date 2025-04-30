from rateslib.fx_volatility.delta_vol import FXDeltaVolSmile, FXDeltaVolSurface
from rateslib.fx_volatility.sabr import FXSabrSmile, FXSabrSurface

__all__ = ["FXSabrSmile", "FXSabrSurface", "FXDeltaVolSurface", "FXDeltaVolSmile"]

FXVols = FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface
FXVolObj = (FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface)
