from typing import Optional, Union, Dict, Any
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import MO, TH, FR

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    next_monday,
    next_monday_or_tuesday,
    sunday_to_monday,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay, Easter, Day, DateOffset
from rateslib.default import NoInput

CalInput = Union[CustomBusinessDay, str, NoInput]

# Generic holidays
Epiphany = Holiday("Epiphany", month=1, day=6)
MaundyThursday = Holiday("Maundy Thursday", month=1, day=1, offset=[Easter(), Day(-3)])
GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])
EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])
AscentionDay = Holiday("Ascention Day", month=1, day=1, offset=[Easter(), Day(39)])
Pentecost = Holiday("PenteCost", month=1, day=1, offset=[Easter(), Day(49)])
WhitMonday = Holiday("Whit Monday", month=1, day=1, offset=[Easter(), Day(50)])
ChristmasEve = Holiday("Christmas Eve", month=12, day=24)
ChristmasDay = Holiday("Christmas Day", month=12, day=25)
ChristmasDayHoliday = Holiday("Christmas Day Holiday", month=12, day=25, observance=next_monday)
ChristmasDayNearestHoliday = Holiday(
    "Christmas Day Sunday Holiday", month=12, day=25, observance=nearest_workday
)
BoxingDay = Holiday("Boxing Day", month=12, day=26)
BoxingDayHoliday = Holiday(
    "Boxing Day Holiday", month=12, day=26, observance=next_monday_or_tuesday
)
NewYearsEve = Holiday("New Year's Eve", month=12, day=31)
NewYearsDay = Holiday("New Year's Day", month=1, day=1)
NewYearsDayHoliday = Holiday("New Year's Day Holiday", month=1, day=1, observance=next_monday)
NewYearsDaySundayHoliday = Holiday(
    "New Year's Day Holiday", month=1, day=1, observance=sunday_to_monday
)
Berchtoldstag = Holiday("Berchtoldstag", month=1, day=2)

# US based
USMartinLutherKingJr = Holiday(
    "Dr. Martin Luther King Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),  # type: ignore[arg-type]
)
USPresidentsDay = Holiday("US President" "s Day", month=2, day=1, offset=DateOffset(weekday=MO(3)))  # type: ignore[arg-type]
USMemorialDay = Holiday("US Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1)))  # type: ignore[arg-type]
USJuneteenthSundayHoliday = Holiday(
    "Juneteenth Independence Day",
    start_date=datetime(2022, 1, 1),
    month=6,
    day=19,
    observance=sunday_to_monday,
)
USIndependenceDayHoliday = Holiday(
    "US Independence Day", month=7, day=4, observance=nearest_workday
)
USLabourDay = Holiday("US Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))  # type: ignore[arg-type]
USColumbusDay = Holiday("US Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2)))  # type: ignore[arg-type]
USVeteransDaySundayHoliday = Holiday("Veterans Day", month=11, day=11, observance=sunday_to_monday)
USThanksgivingDay = Holiday("US Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4)))  # type: ignore[arg-type]

# Canada based
FamilyDay = USPresidentsDay
VictoriaDay = Holiday("Victoria Day", month=5, day=24, offset=DateOffset(weekday=MO(-1)))  # type: ignore[arg-type]
CivicHoliday = Holiday("Civic Holiday", month=8, day=1, offset=DateOffset(weekday=MO(1)))  # type: ignore[arg-type]
CADLabourDay = Holiday("CAD Labour Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))  # type: ignore[arg-type]
CADThanksgiving = Holiday("CAD Thanksgiving", month=10, day=1, offset=DateOffset(weekday=MO(2)))  # type: ignore[arg-type]
Rememberance = Holiday("Rememberance", month=11, day=11, observance=next_monday)
NationalTruth = Holiday(
    "National Truth & Reconciliation", month=9, day=30, start_date=datetime(2021, 1, 1)
)
# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

# UK based
UKEarlyMayBankHoliday = Holiday(
    "UK Early May Bank Holiday", month=5, day=1, offset=DateOffset(weekday=MO(1))  # type: ignore[arg-type]
)
UKSpringBankPre2022 = Holiday(
    "UK Spring Bank Holiday pre 2022",
    end_date=datetime(2022, 5, 1),
    month=5,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
UKSpringBankPost2022 = Holiday(
    "UK Spring Bank Holiday post 2022",
    start_date=datetime(2022, 7, 1),
    month=5,
    day=31,
    offset=DateOffset(weekday=MO(-1)),
)
UKSpringBankHoliday = Holiday(
    "UK Spring Bank Holiday", month=5, day=31, offset=DateOffset(weekday=MO(-1))  # type: ignore[arg-type]
)
UKSummerBankHoliday = Holiday(
    "UK Summer Bank Holiday", month=8, day=31, offset=DateOffset(weekday=MO(-1))  # type: ignore[arg-type]
)

# EUR based
EULabourDay = Holiday("EU Labour Day", month=5, day=1)
SENational = Holiday("Sweden National Day", month=6, day=6)
CHNational = Holiday("Swiss National Day", month=8, day=1)
CADNational = Holiday("Canada Day", month=7, day=1, observance=next_monday)
MidsummerFriday = Holiday("Swedish Midsummer", month=6, day=25, offset=DateOffset(weekday=FR(-1)))  # type: ignore[arg-type]
NOConstitutionDay = Holiday("NO Constitution Day", month=5, day=17)

CALENDAR_RULES: Dict[str, list[Any]] = {
    "bus": [],
    "tgt": [
        NewYearsDay,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        ChristmasDay,
        BoxingDay,
    ],
    "ldn": [
        NewYearsDayHoliday,
        GoodFriday,
        EasterMonday,
        UKEarlyMayBankHoliday,
        UKSpringBankPre2022,
        Holiday("Queen Jubilee Thu", year=2022, month=6, day=2),
        Holiday("Queen Jubilee Fri", year=2022, month=6, day=3),
        Holiday("Queen Funeral", year=2022, month=9, day=19),
        UKSpringBankPost2022,
        Holiday("King Charles III Coronation", year=2023, month=5, day=8),
        UKSummerBankHoliday,
        ChristmasDayHoliday,
        BoxingDayHoliday,
    ],
    "nyc": [
        NewYearsDaySundayHoliday,
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        USJuneteenthSundayHoliday,
        USIndependenceDayHoliday,
        USLabourDay,
        USColumbusDay,
        USVeteransDaySundayHoliday,
        USThanksgivingDay,
        ChristmasDayNearestHoliday,
        Holiday("GHW Bush Funeral", year=2018, month=12, day=5),
    ],
    "stk": [
        NewYearsDay,
        Epiphany,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        AscentionDay,
        SENational,
        MidsummerFriday,
        ChristmasEve,
        ChristmasDay,
        BoxingDay,
        NewYearsEve,
    ],
    "osl": [
        NewYearsDay,
        MaundyThursday,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        NOConstitutionDay,
        AscentionDay,
        WhitMonday,
        ChristmasEve,
        ChristmasDay,
        BoxingDay,
    ],
    "zur": [
        NewYearsDay,
        Berchtoldstag,
        GoodFriday,
        EasterMonday,
        EULabourDay,
        AscentionDay,
        WhitMonday,
        CHNational,
        # ChristmasEve,
        ChristmasDay,
        BoxingDay,
        # NewYearsEve,
    ],
    "tro": [
        NewYearsDayHoliday,
        FamilyDay,
        GoodFriday,
        VictoriaDay,
        CADNational,
        CivicHoliday,
        CADLabourDay,
        NationalTruth,
        CADThanksgiving,
        Rememberance,
        ChristmasDayHoliday,
        BoxingDayHoliday,
    ],
}


def create_calendar(rules: list, weekmask: Optional[str] = None) -> CustomBusinessDay:
    """
    Create a calendar with specific business and holiday days defined.

    Parameters
    ----------
    rules : list[Holiday]
        A list of specific holiday dates defined by the
        ``pandas.tseries.holiday.Holiday`` class.
    weekmask : str, optional
        Set of days as business days. Defaults to *"Mon Tue Wed Thu Fri"*.

    Returns
    --------
    CustomBusinessDay

    Examples
    --------
    .. ipython:: python

       from pandas.tseries.holiday import Holiday
       from pandas import date_range
       TutsBday = Holiday("Tutankhamum Birthday", month=7, day=2)
       pyramid_builder = create_calendar(rules=[TutsBday], weekmask="Tue Wed Thu Fri Sat Sun")
       construction_days = date_range(dt(1999, 6, 25), dt(1999, 7, 5), freq=pyramid_builder)
       construction_days

    """
    weekmask = "Mon Tue Wed Thu Fri" if weekmask is None else weekmask
    return CustomBusinessDay(  # type: ignore[call-arg]
        calendar=AbstractHolidayCalendar(rules=rules),
        weekmask=weekmask,
    )


# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


CALENDARS: Dict[str, CustomBusinessDay] = {
    "bus": create_calendar(rules=CALENDAR_RULES["bus"], weekmask="Mon Tue Wed Thu Fri"),
    "tgt": create_calendar(rules=CALENDAR_RULES["tgt"], weekmask="Mon Tue Wed Thu Fri"),
    "ldn": create_calendar(rules=CALENDAR_RULES["ldn"], weekmask="Mon Tue Wed Thu Fri"),
    "nyc": create_calendar(rules=CALENDAR_RULES["nyc"], weekmask="Mon Tue Wed Thu Fri"),
    "stk": create_calendar(rules=CALENDAR_RULES["stk"], weekmask="Mon Tue Wed Thu Fri"),
    "osl": create_calendar(rules=CALENDAR_RULES["osl"], weekmask="Mon Tue Wed Thu Fri"),
    "zur": create_calendar(rules=CALENDAR_RULES["zur"], weekmask="Mon Tue Wed Thu Fri"),
    "tro": create_calendar(rules=CALENDAR_RULES["tro"], weekmask="Mon Tue Wed Thu Fri"),
}

### RUN THE SCRIPT TO EXPORT HOLIDAY LIST
cal = CALENDARS["tgt"]
ts = pd.to_datetime(cal.holidays)
strings = ['"'+_.strftime("%Y-%m-%d %H:%M:%S")+'"' for _ in ts]
line = ",\n".join(strings)
print(line)
