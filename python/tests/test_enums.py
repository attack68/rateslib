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

from rateslib.enums import FloatFixingMethod


def test_method_param():
    a1 = FloatFixingMethod.RFRPaymentDelay()
    a2 = FloatFixingMethod.RFRPaymentDelayAverage()

    for obj in [a1, a2]:
        assert obj.method_param() == 0

    b1 = FloatFixingMethod.IBOR(6)
    b2 = FloatFixingMethod.RFRLookback(6)

    for obj in [b1, b2]:
        assert obj.method_param() == 6
