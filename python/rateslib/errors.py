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

# Arg Parsing

VE_NEEDS_FREQUENCY = "`frequency` as string or Frequency is needed to perform tenor calculations."

VE_NEEDS_FIXEDRATE = "A `fixed_rate` must be set for a cashflow to be determined."

VE_ATTRIBUTE_IS_IMMUTABLE = (
    "The '{}' attribute is immutable to avoid conflicting calculations. Re-initialize the instance."
)

VE_ND_LEG_NEEDS_NO_EXCHANGES = (
    "An Leg defined as non-deliverable by some parameter, e.g. `pair` cannot have "
    "notional exchanges."
)

VE_PAIR_AND_LEG_MTM = "Setting `mtm` on a Leg requires a non-deliverable `pair` input."

# Curve Parsing

NI_NO_DISC_FROM_DICT = "`disc_curve` cannot currently be parsed from a dictionary of curves."

VE_NEEDS_DISC_CURVE = (
    "`disc_curve` is required but it has not been provided, or cannot be parsed from an external "
    "`curves` argument."
)

VE_NO_DISC_FROM_VALUES = "`disc_curve` cannot be inferred from a non-DF based curve."

VE_BEFORE_INITIAL = "The Curve initial node date is after the required forecasting date."

# Period Parameters

VE_NEEDS_INDEX_PARAMS = (
    "`{0}` must be initialised with index parameters, i.e. those for `_IndexParams`. See docs."
)

VE_HAS_INDEX_PARAMS = (
    "`{0}` must not be initialised with index parameters, i.e. those for `_IndexParams`. See docs."
)

VE_NEEDS_ND_CURRENCY_PARAMS = (
    "`{0}` must be initialised with non-deliverable currency parameters, i.e. those for "
    "`_CurrencyParams`. See docs."
)

VE_HAS_ND_CURRENCY_PARAMS = (
    "`{0}` must not be initialised with non-deliverable currency parameters, i.e. those for "
    "`_CurrencyParams`. See docs."
)

VE_MISMATCHED_FX_PAIR_ND_PAIR = (
    "Non-deliverable FXOptions into a third currency are not allowed.\n"
    "Got nd-currency: '{0}' and option index pair: '{1}'.\n"
    "FXOptions of this nature require quanto volatility adjustements that the basic models"
    "do not include."
)

# Fixings

UW_NO_TENORS = (
    "The IBORStubFixing has not detected any tenors under the identifier: '{0}' and "
    "will therefore never obtain any fixing value."
)

TE_NO_FIXING_EXPOSURE_ON_OBJ = (
    "The object type '{0}' does not contain or have available methods to calculate fixings "
    "exposure."
)

VE01_1 = (
    "Fixing data for the index '{0}' has been attempted, but none found.\nEither there "
    "is no data file ('{0}.csv') located in the searched data directory,\nor a Series "
    "has not been added manually by performing `fixings.add"
    "('{0}', some_series)`.\nTo create a CSV file in the searched data directory "
    "use the exact template structure for the file between the hashes:\n"
    "###################\n"
    "reference_date,rate\n26-08-2023,5.6152\n27-08-2023,5.6335\n##################\n"
    "For further info see 'Working with Fixings' in the documentation cookbook.",
)

AE_NEEDS_PAIR_TO_FORECAST = (
    "A currency `pair` is required for non-deliverable `fx_fixing` forecasting."
)

VE_NEEDS_FX_FORWARDS = (
    "An FXForwards object for `fx` is required for instrument pricing.\n"
    "If this instrument is part of a Solver, have you omitted the `fx` input?",
)

VE_NEEDS_FX_FORWARDS_BAD_TYPE = (
    "An FXForwards object for `fx` is required for instrument pricing.\n"
    "The given type, '{0}', cannot be used here."
)

FW_FIXINGS_AS_SERIES = (
    "Setting any `fixings` argument as a Series directly is currently supported, but not "
    "recommended and may be removed in future versions.\n"
    "Best practice is to add the fixings object to the default _BaseFixingsLoader and then "
    "reference that object by Series name.\n"
    "For example, change: `rate_fixings`=my_series_object` to\n"
    "`fixings.add('EURIBOR_3M', my_series_object)`\n"
    "`fixings.add('EURIBOR_6M', another_series_object)`\n"
    "`rate_fixings='EURIBOR'`\n"
    "See cookbook article 'Working with Fixings' for more information."
)

VE_INDEX_FIXINGS_AS_STR_OR_VALUE = (
    "`index_fixings` must be specified either as a scalar value or a string identifier for a "
    "fixings set in the _BaseFixingsLoader. Got type: {0}."
)

VE_INDEX_LAG_MUST_BE_ZERO = (
    "`index_lag` must be zero when using a 'Curve' `index_method`.\n"
    "`index_date`: {0}, is in Series but got `index_lag`: {1}."
)

VE_EMPTY_SERIES = "An fixing value cannot be derived from an `fixings` Series having no entries."

VE_INDEX_BASE_NO_STR = (
    "`index_base` argument cannot be initialised as string.\n If seeking to determine its "
    "value with a Fixings series then do not provide any `index_base` value and use "
    "`index_fixings` instead.\nOr use the 'index_value' method to separately determine a "
    "scalar value to enter directly as the `index_base` argument."
)

# VE_NEEDS_INDEX_BASE_DATE = (
#     "An `index_base` forecast value requires an `index_base_date` to be provided."
# )

# 08: periods/components/parameters.py

VE08_0 = (
    "The `index_base` is not an explicitly provided value for the Period.\n"
    "`index_base_date` must therefore be provided to forecast `index_base` from an `index_curve` "
    "or `index_fixings`."
)

VE08_1 = (
    "Must supply an `index_date` from which to forecast if `index_fixings` is not provided.\n"
    "This error usually arises when an `index_base` value is not provided for a Period and "
    "there is no `index_base_date`,\nor if there are no `index_fixings` and there is no "
    "`index_reference_date` is combination."
)


VE_NEEDS_STRIKE = "An FXOptionPeriod cashflow cannot be determined without setting a `strike`."


# VE_NEEDS_FIXING_SERIES = (
#     "A `fixing_series` must be supplied for floating rate parameters."
# )

# VE_NEEDS_FIXING_FREQUENCY = "A `fixing_frequency` must be supplied for floating rate parameters."

# 02: periods/components/float_rate.py

VE_NEEDS_RATE_CURVE = "A `rate_curve` must be provided to this method."

VE_MISMATCHED_ND_PAIR = (
    "A non-deliverable pair must contain the settlement currency.\nGot '{0}' and '{1}'."
)

MISMATCH_RATE_INDEX_PARAMETERS = (
    "A `rate_curve` and `rate_index` have been supplied with conflicting parameters.\n"
    "Specifically for the attribute '{0}'\n"
    "Got: '{1}' and '{2}'."
)

VE_NEEDS_CURVE_OR_INDEX = (
    "Either `rate_curve` or `rate_index` must be provided so that the "
    "conventions for the floating rate, such as the fixing calendar and the accrual "
    "convention can be determined."
)

VE_NEEDS_RATE_TO_FORECAST_RFR = (
    "A `rate_curve` is required to forecast missing RFR rates.\n"
    "This may be observed as a direct argument input, or this error may by a result of "
    "incorrectly supplying the `curves` argument to any Instrument class."
)

VE_NEEDS_RATE_TO_FORECAST_STUB_IBOR = (
    "A `rate_curve` is required to forecast missing IBOR rate.\n"
    "`rate_curve` might be specifically omitted or an external `curves` argument may be "
    "malformed.\nNote that forecasting an IBOR stub from a single curve is bad practice and "
    "a more accurate calculation will likely be obtained from a dict of curves, e.g.\n"
    "'{'1m': curve1, '3m': curve2, '6m': curve3}'"
)

VE_NEEDS_RATE_TO_FORECAST_TENOR_IBOR = (
    "A `rate_curve` is required to forecast missing IBOR rate.\n"
    "`rate_curve` might be specifically omitted or an external `curves` argument may be "
    "malformed."
)

VE_FIXINGS_BAD_TYPE = (
    "`.._fixings` should be a single value or a string labelling a fixing set in the "
    "`fixings` container. It cannot be a list or Series.\n"
    "To migrate from the legacy implementation where a Series could be supplied directly "
    "use the following:\nAdd your Series to defaults: `default.fixings.add('EURIBOR_3M', "
    "my_series_obj)`\nAnd then reference this fixing set directly: `rate_fixings='EURIBOR'`.\nThe"
    "suffix '_3M' will be added directly internally (based on the Frequency) and will adjust for "
    "stub fixings. RFR fixings will have the '_1B' suffix added, so use for example:\n"
    "Add an RFR Series: `fixings.add('SOFR_1B', my_series_obj)`\n"
    "And reference this set directly: `rate_fixings='SOFR'`.\n"
    "For further details see the cookbook documentation entitled 'Working with Fixings'."
)

VE02_1 = (
    "RFR Observation and Accrual DCF dates do not align.\nThis is usually the result of a "
    "'rfr_lookback' Period which does not adhere to the holiday calendar of the `curve`.\n"
    "start date: {0} is curve holiday? => {1}\nend date: {2} is curve holiday? => {3}\n"
)
VE02_2 = (
    "The accrual `start` and `end` dates ({0} and/or {1}) for the period do not align with "
    "business days under the `fixing_calendar`.\nRFR Periods need to align with valid fixing"
    "days."
)
VE02_3 = (
    "Providing `rate_fixings` as a scalar value for an RFR type `fixing_method` is not "
    "permitted due to ambiguity, particularly in combination with the `float_spread`.\n"
    "Consider adding a Series to `defaults`: `fixings.add('MY_RFR_1B', "
    "some_series)`\nAnd then referencing this fixings collection: `rate_fixings='MY_RFR'\n"
    "For an RFR type fixing method the suffix added internally is always '_1B'."
)

VE_SPREAD_METHOD_RFR = (
    "The `spread_compound_method` must be the 'NoneSimple' variant when using a "
    "`fixing_method` which defines an RFR Averaging type calculation.\nGot: {0}"
)

VE02_5 = (
    "The fixings series '{0}' for the RFR 1B rates is missing a value expected by the fixings "
    "calendar.\n"
    "Specifically '{1}' is expected, yet '{2}' is provided implying a data entry is missing."
)

VE_NEEDS_RATE_POPULATE_FIXINGS = (
    "A `rate_curve` is required to forecast missing RFR fixings in a floating rate calculation.\n"
    "This may be a direct input or the input to an Instrument's `curves` argument may be incorrect."
    "\nThe missing data is shown below for this calculation:\n"
    "{0}"
)

VE_LOCKOUT_METHOD_PARAM = (
    "The `method_param` for an RFR Lockout type `fixing_method` must not exceed the length of the "
    "period.\nGot: '{0}' for the following fixing rates:\n{1}"
)

W02_0 = (
    "The fixings series '{0}' for the RFR 1B rates contains more fixings than are expected from "
    "the fixings calendar.\n"
    "Specifically, the extra data item lies within the fixings window: '{1}':'{2}'."
)
