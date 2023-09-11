import pandas as pd
import os
from packaging import version


def _append_kwargs_name(df):
    """combine the columns leg and kwargs to produce library consistent kwargs for dicts"""
    prefix = df["leg"]
    prefix = prefix.where(prefix == "leg2", "")
    prefix = prefix.replace("leg2", "leg2_")
    df["kwargs_name"] = prefix + df["kwarg"]
    return df.set_index("kwargs_name")


def _parse_bool(df):
    """parse data input as bools to return True and False dtypes."""

    def _map_true_false(v):
        try:
            if v.upper() == "TRUE":
                return True
            elif v.upper() == "FALSE":
                return False
        except AttributeError:
            return None
        else:
            return None

    if version.parse(pd.__version__) >= version.parse("2.1.0"):
        # applymap issues a deprecation warning with version <2.1.0
        # TODO (low): clean this up when setting a minimum pandas version at 2.1.0
        df[df["dtype"] == "bool"] = df[df["dtype"] == "bool"].map(_map_true_false)
    else:
        df[df["dtype"] == "bool"] = df[df["dtype"] == "bool"].applymap(_map_true_false)
    return df


path = "data/instrument_spec.csv"
abspath = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(abspath, path)
df = pd.read_csv(target)
df = _append_kwargs_name(df)
df = _parse_bool(df)
df_legs = df[~(df["leg"] == "meta")]

DTYPE_MAP = {
    "str": str,
    "float": float,
    "bool": bool,
    "int": int,
}


def _map_dtype(v):
    try:
        return DTYPE_MAP[v]
    except KeyError:
        return v


def _map_str_int(v):
    try:
        return int(v)
    except ValueError:
        return v


def _get_kwargs(spec):
    """From the legs DataFrame extract the relevant column and ensure dtypes are suitable."""
    # get values that are not null
    s = df_legs[spec]
    s = s[pd.notna(s)]
    # assign the correct dtypes for the values
    dfs = s.to_frame().transpose()
    dtypes = df.loc[s.index, "dtype"]
    dtypes = dtypes.map(_map_dtype)
    dfs = dfs.astype(dtype=dtypes.to_dict(), errors="raise")
    # rotate and return values in a dict
    s = dfs.transpose()[spec]
    d = s.to_dict()

    # roll dtype is str or int causes object issues
    if "roll" in d:
        d["roll"] = _map_str_int(d["roll"])
    if "leg2_roll" in d:
        d["leg2_roll"] = _map_str_int(d["leg2_roll"])
    return d


INSTRUMENT_SPECS = {
    "usd_irs": _get_kwargs("usd_irs"),  # IRS
    "gbp_irs": _get_kwargs("gbp_irs"),
    "eur_irs": _get_kwargs("eur_irs"),
    "eur_irs3": _get_kwargs("eur_irs3"),
    "eur_irs6": _get_kwargs("eur_irs6"),
    "sek_irs": _get_kwargs("sek_irs"),
    "sek_irs3": _get_kwargs("sek_irs3"),
    "nok_irs": _get_kwargs("nok_irs"),
    "nok_irs3": _get_kwargs("nok_irs3"),
    "nok_irs6": _get_kwargs("nok_irs6"),
    "chf_irs": _get_kwargs("chf_irs"),
    "eur_fra3": _get_kwargs("eur_fra3"),  # FRA
    "eur_fra6": _get_kwargs("eur_fra6"),
    "sek_fra3": _get_kwargs("sek_fra3"),
    "eur_sbs36": _get_kwargs("eur_sbs36"),  # SBS
    "eurusd_xcs": _get_kwargs("eurusd_xcs"),  # XCS
    "gbpusd_xcs": _get_kwargs("gbpusd_xcs"),
    "eurgbp_xcs": _get_kwargs("eurgbp_xcs"),
    "eur_zcis": _get_kwargs("eur_zcis"),  # ZCIS
    "gbp_zcis": _get_kwargs("gbp_zcis"),
    "usd_zcis": _get_kwargs("usd_zcis"),
    "gbp_zcs": _get_kwargs("gbp_zcs"),  # ZCS
    "sek_iirs": _get_kwargs("sek_iirs"),  # IIRS
    "usd_gb": _get_kwargs("usd_gb"),  # FRB
    "usd_gbb": _get_kwargs("usd_gbb"),
    "gbp_gb": _get_kwargs("gbp_gb"),
    "cad_gb": _get_kwargs("cad_gb"),
    "sek_gb": _get_kwargs("sek_gb"),
    "sek_gbb": _get_kwargs("sek_gbb"),
    "usd_frn5": _get_kwargs("usd_frn5"),  # FRN
    "test": _get_kwargs("test"),  # TEST
}

# add some aliases

INSTRUMENT_SPECS = {
    **INSTRUMENT_SPECS,
    "sofr": INSTRUMENT_SPECS["usd_irs"],
    "ust": INSTRUMENT_SPECS["usd_gb"],
    "ustb": INSTRUMENT_SPECS["usd_gbb"],
    "ukt": INSTRUMENT_SPECS["gbp_gb"],
    "gilt": INSTRUMENT_SPECS["gbp_gb"],
    "cadgb": INSTRUMENT_SPECS["cad_gb"],
    "sgb": INSTRUMENT_SPECS["sek_gb"],
    "sgbb": INSTRUMENT_SPECS["sek_gbb"],
}
