import pandas as pd

from api.data import compute_monthly_sales, tag_outliers


def test_compute_monthly_sales():
    df = pd.DataFrame(
        columns=["year_week", "vegetable", "sales"],
        data=[
            [202003, "tomato", 10],
            [202004, "tomato", 700],
        ],
    )

    df_expected = pd.DataFrame(
        columns=["year_month", "vegetable", "sales"],
        data=[[202001, "tomato", 510], [202002, "tomato", 200]],
    )

    df_res = compute_monthly_sales(df)

    pd.testing.assert_frame_equal(df_res, df_expected, check_dtype=False)  # type: ignore


def test_tag_outliers():
    df = pd.DataFrame(
        columns=["year_week", "vegetable", "sales"],
        data=[
            [202001, "tomato", 10],
            [202002, "tomato", 10],
            [202003, "tomato", 10],
            [202004, "tomato", 10],
            [202005, "tomato", 10],
            [202006, "tomato", 10],
            [202007, "tomato", 10],
            [202008, "tomato", 10],
            [202009, "tomato", 10],
            [202010, "tomato", 10_000],
            [202011, "tomato", 10],
            [202012, "tomato", 10],
            [202013, "tomato", 10],
            [202014, "tomato", 10],
            [202015, "tomato", 10],
            [202016, "tomato", 10],
            [202017, "tomato", 10],
            [202018, "tomato", 10],
            [202019, "tomato", 10],
            [202021, "tomato", 10],
            [202022, "tomato", 10],
            [202023, "tomato", 10],
            [202024, "tomato", 10],
            [202025, "tomato", 10],
            [202026, "tomato", 10],
            [202027, "tomato", 10],
            [202028, "tomato", 10],
            [202029, "tomato", 10],
        ],
    )

    df_expected = pd.DataFrame(
        columns=["year_week", "vegetable", "sales", "is_outlier"],
        data=[
            [202001, "tomato", 10, False],
            [202002, "tomato", 10, False],
            [202003, "tomato", 10, False],
            [202004, "tomato", 10, False],
            [202005, "tomato", 10, False],
            [202006, "tomato", 10, False],
            [202007, "tomato", 10, False],
            [202008, "tomato", 10, False],
            [202009, "tomato", 10, False],
            [202010, "tomato", 10_000, True],
            [202011, "tomato", 10, False],
            [202012, "tomato", 10, False],
            [202013, "tomato", 10, False],
            [202014, "tomato", 10, False],
            [202015, "tomato", 10, False],
            [202016, "tomato", 10, False],
            [202017, "tomato", 10, False],
            [202018, "tomato", 10, False],
            [202019, "tomato", 10, False],
            [202021, "tomato", 10, False],
            [202022, "tomato", 10, False],
            [202023, "tomato", 10, False],
            [202024, "tomato", 10, False],
            [202025, "tomato", 10, False],
            [202026, "tomato", 10, False],
            [202027, "tomato", 10, False],
            [202028, "tomato", 10, False],
            [202029, "tomato", 10, False],
        ],
    )

    df_res = tag_outliers(df)

    pd.testing.assert_frame_equal(df_res, df_expected)
