import pandas as pd

from src.data.news import align_to_trading_day

# Mon 2021-03-15 .. Fri 2021-03-19, Mon 2021-03-22 (EDT, UTC-4)
CAL = pd.DatetimeIndex(["2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-22"])


def _td(stamps, **kw):
    out = align_to_trading_day(pd.Series(pd.to_datetime(stamps)), CAL, **kw)
    return [None if pd.isna(x) else x.strftime("%Y-%m-%d") for x in out.trading_day]


def test_before_cutoff_same_day():
    # 19:59 UTC = 15:59 EDT -> same day
    assert _td(["2021-03-16 19:59:00"]) == ["2021-03-16"]


def test_at_or_after_cutoff_next_day():
    # 20:00 UTC = 16:00 EDT -> next trading day; 23:30 UTC also next day
    assert _td(["2021-03-16 20:00:00", "2021-03-16 23:30:00"]) == ["2021-03-17", "2021-03-17"]


def test_utc_after_midnight_is_previous_et_evening():
    # 2021-03-17 01:00 UTC = 2021-03-16 21:00 EDT -> after close on the 16th -> the 17th
    assert _td(["2021-03-17 01:00:00"]) == ["2021-03-17"]


def test_weekend_rolls_to_monday():
    assert _td(["2021-03-20 15:00:00", "2021-03-21 12:00:00"]) == ["2021-03-22", "2021-03-22"]


def test_friday_after_close_goes_to_monday():
    assert _td(["2021-03-19 21:00:00"]) == ["2021-03-22"]


def test_date_only_conservative_next_day():
    # 00:00:00 means the time is unknown: treat it as after the close of that calendar date
    assert _td(["2021-03-16 00:00:00"]) == ["2021-03-17"]
    assert _td(["2021-03-16 00:00:00"], date_only_policy="same_day") == ["2021-03-16"]


def test_beyond_calendar_is_nat():
    assert _td(["2021-03-22 21:00:00"]) == [None]


def test_eastern_source_timezone():
    # If the source were already ET, 15:30 stays on the same day
    assert _td(["2021-03-16 15:30:00"], src_tz="America/New_York") == ["2021-03-16"]
