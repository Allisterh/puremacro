import pandas as pd

from puremacro._codes import drop_aggregates, is_country


def test_is_country() -> None:
    # Valid countries
    assert is_country("USA")
    assert is_country("GBR")
    assert is_country("aus") # should convert upper
    assert is_country("MEX")

    # Invalid strings
    assert not is_country("US")     # length < 3
    assert not is_country("USAA")   # length > 3
    assert not is_country("U1A")    # not alpha
    assert not is_country("")       # empty

    # Aggregates
    assert not is_country("WLD")
    assert not is_country("EA20")
    assert not is_country("OECDE")
    assert not is_country("EMU")
    assert not is_country("world")

    # Non-strings
    assert not is_country(None)
    assert not is_country(123)
    assert not is_country(["USA"])

def test_drop_aggregates() -> None:
    df = pd.DataFrame({
        "code": ["USA", "WLD", "GBR", "EA20", "MEX", "US", "123"],
        "value": [1, 2, 3, 4, 5, 6, 7]
    })

    result = drop_aggregates(df)

    expected = pd.DataFrame({
        "code": ["USA", "GBR", "MEX"],
        "value": [1, 3, 5]
    })

    pd.testing.assert_frame_equal(result, expected)

def test_drop_aggregates_custom_column() -> None:
    df = pd.DataFrame({
        "country_code": ["USA", "WLD", "GBR", "EA20", "MEX"],
        "value": [1, 2, 3, 4, 5]
    })

    result = drop_aggregates(df, code_col="country_code")

    expected = pd.DataFrame({
        "country_code": ["USA", "GBR", "MEX"],
        "value": [1, 3, 5]
    })

    pd.testing.assert_frame_equal(result, expected)

def test_drop_aggregates_empty_df() -> None:
    df = pd.DataFrame(columns=["code", "value"])
    result = drop_aggregates(df)
    pd.testing.assert_frame_equal(result, df)

def test_drop_aggregates_all_aggregates() -> None:
    df = pd.DataFrame({
        "code": ["WLD", "EA20", "OECDE"],
        "value": [1, 2, 3]
    })
    result = drop_aggregates(df)
    expected = df.iloc[:0].reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)
