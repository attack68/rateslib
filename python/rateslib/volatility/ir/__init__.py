# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################


from rateslib.volatility.ir.sabr import IRSabrCube, IRSabrSmile

__all__ = [
    "IRSabrSmile",
    "IRSabrCube",
]

IRVols = IRSabrSmile | IRSabrCube
IRVolObj = (IRSabrSmile, IRSabrCube)
