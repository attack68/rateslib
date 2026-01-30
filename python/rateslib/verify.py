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

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
    )

VERSION = "2.6.0"


class LicenceNotice(UserWarning):
    _no_licence_warning = (
        "\nRateslib is source-available (not open-source) software distributed under a "
        "dual-licence model."
        "\nNo commercial licence is registered for this installation. Use is therefore permitted "
        "for non-commercial purposes only (at-home or university based academic use)."
        "\nAny use in commercial, professional, or for-profit environments, including evaluation "
        "or trial use, requires a valid commercial licence or an approved evaluation licence."
        "\nCertain features may require a registered commercial or evaluation licence in current "
        "or future versions."
        "\nFor licensing information or to register a licence, please visit: "
        "https://rateslib.com/licence"
    )

    _invalid_warning = (
        "\nRateslib is source-available (not open-source) software distributed under a "
        "dual-licence model."
        "\nAn invalid licence file is detected for this installation. Use is therefore permitted "
        "for non-commercial purposes only (at-home or university based academic use)."
        "\nAny use in commercial, professional, or for-profit environments, including evaluation "
        "or trial use, requires a valid commercial licence or an approved evaluation licence."
        "\nCertain features may require a registered commercial or evaluation licence in current "
        "or future versions."
        "\nFor licensing information or to register a licence, please visit: "
        "https://rateslib.com/licence"
    )

    _expired_warning = (
        "\nYour existing licence for rateslib {0} expired on {1}.\n"
        "If you wish to extend your licence, please visit https://rateslib.com/licence for further "
        "details.\n"
        "Otherwise, please uninstall rateslib.\n"
        "Expired licence details:\n{2}\n"
    )


class _LicenceStatus(Enum):
    VALID = 0
    EXPIRED_GRACE = 1
    EXPIRED = 2
    INVALID = 3
    NO_LICENCE = 4


class Licence:
    """
    A licence coordinator to control warnings and functionality.
    """

    def __init__(self) -> None:
        # search for licences in relevant paths
        value = os.getenv("RATESLIB_LICENCE") or _get_licence()

        if value is None:
            # then no licence data was found either in environment vars or in the standard path.
            self._status = _LicenceStatus.NO_LICENCE
        else:
            verified_expiry = _verify_licence(value)

            if verified_expiry is None:  # i.e. invalid signature key
                self._status = _LicenceStatus.INVALID
            else:
                # measure the expiry relative to today
                self._expiry = datetime.strptime(verified_expiry, "%Y-%m-%d")
                if self.expiry > datetime.now():
                    self._status = _LicenceStatus.VALID
                elif self.expiry > datetime.now() - timedelta(days=14):
                    self._status = _LicenceStatus.EXPIRED_GRACE
                else:
                    self._status = _LicenceStatus.EXPIRED

        if self.status == _LicenceStatus.NO_LICENCE:
            self._output(LicenceNotice._no_licence_warning, VERSION)
        elif self.status == _LicenceStatus.INVALID:
            self._output(LicenceNotice._invalid_warning, VERSION)
        elif self.status == _LicenceStatus.EXPIRED:
            self._output(
                LicenceNotice._expired_warning,
                VERSION,
                self.expiry.strftime("%Y-%m-%d"),
                self.print_licence(),
            )

    def _output(self, text: str, *args: Any) -> None:
        warnings.warn(message=text.format(*args), category=LicenceNotice, stacklevel=4)
        logger = logging.getLogger(__name__)
        logger.info(text.format(*args))

    @property
    def status(self) -> _LicenceStatus:
        return self._status

    @property
    def expiry(self) -> datetime:
        return self._expiry

    @classmethod
    def add_licence(cls, licence_text: str) -> None:
        """
        Store the provided licence as a file on the local disk.

        Will create or overwrite any existing licence file as necessary. Will raise
        PermissionError if writing to disk fails due to restrictions.

        Parameters
        ----------
        licence_text: str
            The full JSON format str of the provided licence.

        Returns
        -------
        None
        """
        licence_file = _get_licence_path()
        try:
            if licence_file.exists():
                current = licence_file.read_text()
                if current != licence_text:
                    print(f"Warning: Existing licence differs. Overwriting {licence_file}")

            # only add if a valid licence string:
            try:
                valid = _verify_licence(licence_text)
            except JSONDecodeError:
                raise ValueError(
                    "The provided licence text does not appear to be valid JSON format or cannot "
                    f"be decoded as such.\n{licence_text}"
                )

            if not valid:
                raise ValueError(
                    f"The licence key is invalid and has not been saved to disk.\n{licence_text}"
                )

            licence_file.write_text(licence_text)
            print(f"License saved at {licence_file}")
        except PermissionError:
            raise PermissionError(
                f"Cannot save licence file to {licence_file}.\n "
                f"Check your admin or corporate file permissions."
            )

    @classmethod
    def remove_licence(cls) -> bool:
        """
        Remove the stored licence file.

        Raises PermissionError if the file cannot be deleted from disk due to restrictions.

        Returns
        -------
        bool
            *True* on successful removal and *False* if no licence file exists.
        """
        licence_file = _get_licence_path()
        try:
            if licence_file.exists():
                licence_file.unlink()
                print(f"License removed from {licence_file}")
                return True
            else:
                print("No licence file found to remove.")
                return False
        except PermissionError:
            raise PermissionError(
                f"Cannot remove licence file at {licence_file}. Check your permissions."
            )

    @classmethod
    def print_licence(cls) -> str:
        """
        Output the licence data to string.

        Returns
        -------
        str
            The JSON format of the licence.
        """
        value = os.getenv("RATESLIB_LICENCE") or _get_licence()
        if value is None:
            raise ValueError("No rateslib licence data was found on this machine")
        else:
            return value


APP_NAME = "rateslib"
LICENSE_FILENAME = "rateslib_licence.txt"


PUBLIC_KEY: tuple[int, int] = (
    65537,
    86222696103896966718103037502072442336246185093318724988310224539490986842962518392592510336894335238460512594559929385462044884137775548353223089347652775415882082908041940084967476300969806363550378972881577687292674787317782507726743027399965228306794174501671206473081788525064813988527838836758351217651,
)


def _rsa_encrypt(message_int: int, public_key: tuple[int, int]) -> int:
    e, n = public_key
    if not 0 <= message_int < n:
        raise ValueError("Message too large for key")
    return pow(message_int, e, n)


def _get_licence_path() -> Path:
    """
    Returns the path where the licence file should be stored.
    Cross-platform user-specific location.
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux / Unix
        base = Path.home() / ".local" / "share"

    data_dir = base / APP_NAME
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    return data_dir / LICENSE_FILENAME


def _get_licence() -> str | None:
    """
    Retrieve the stored licence text, or None if not found.
    """
    licence_file = _get_licence_path()
    if licence_file.exists():
        return licence_file.read_text()
    return None


def _verify_licence(licence_plaintext: str) -> str | None:
    loaded_dict = json.loads(licence_plaintext)
    licence_dict = dict(sorted(loaded_dict.items()))

    hex_s = licence_dict.pop("xkey", None)
    if hex_s is None:
        return None

    s = int(hex_s, 16)

    m = json.dumps(licence_dict, sort_keys=True)
    hex_h = hashlib.sha256(m.encode()).hexdigest()
    h = int(hex_h, 16)  # h = int.from_bytes(hashlib.sha256(m.encode()).digest())

    h_ = _rsa_encrypt(s, PUBLIC_KEY)

    if h != h_:
        return None
    else:
        try:
            return loaded_dict["expiry"]  # type: ignore[no-any-return]
        except KeyError:
            return None


__all__ = ["LicenceNotice", "Licence"]
