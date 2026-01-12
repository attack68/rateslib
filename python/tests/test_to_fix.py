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

from datetime import datetime as dt

import pytest
from rateslib.dual import Dual
from rateslib.fx_volatility import FXDeltaVolSmile


def test_fxsmile_update_node():
    # update node does not validate the AD order of the supplied value
    # this should probably return a more helpful error message
    fxs = FXDeltaVolSmile(
        eval_date=dt(2000, 1, 1),
        expiry=dt(2000, 12, 1),
        nodes={0.1: 1, 0.2: 2},
        delta_type="forward",
    )
    fxs._set_ad_order(2)
    with pytest.raises(TypeError):
        fxs.update_node(0.1, Dual(2.0, ["x"], []))
