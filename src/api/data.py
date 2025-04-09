import pandas as pd


def clean_vegetable_name(df):
    mapping = {
        "Tomatto": "tomato",
        "Tomate": "tomato",
    }
    df["vegetable"] = df["vegetable"].replace(mapping)
    return df.groupby(["year_week", "vegetable"], as_index=False)["sales"].sum()


def compute_monthly_sales(df):
    df["last_day"] = pd.to_datetime(10 * df["year_week"] + 0, format="%Y%W%w")
    df["first_day"] = pd.to_datetime(10 * df["year_week"] + 1, format="%Y%W%w")
    df["year_month_of_last_day"] = df["last_day"].dt.month
    df["days_next_month"] = df["last_day"].dt.day.clip(upper=7)
    df["days_prev_month"] = 7 - df["days_next_month"]

    df["sales_daily"] = df["sales"] / 7

    df_next_month = df[["last_day", "vegetable", "sales_daily", "days_next_month"]]
    df_next_month["sales"] = df["sales_daily"] * df["days_next_month"]
    df_next_month["year_month"] = df["last_day"].dt.strftime("%Y%m")

    df_prev_month = df.query("days_prev_month > 0")[
        ["first_day", "vegetable", "sales_daily", "days_prev_month"]
    ]
    df_prev_month["sales"] = df["sales_daily"] * df["days_prev_month"]
    df_prev_month["year_month"] = df["first_day"].dt.strftime("%Y%m")

    df_result = (
        pd.concat([df_next_month, df_prev_month])
        .groupby(["year_month", "vegetable"], as_index=False)["sales"]
        .sum()
    )
    df_result["year_month"] = df_result["year_month"].astype("int64")
    return df_result


def tag_outliers(df, cut_stdev=5):
    df_stat = df.groupby("vegetable").agg({"sales": ("mean", "std")})
    df_stat.columns = ["mean", "std"]
    df_stat["sales_min"] = df_stat["mean"] - cut_stdev * df_stat["std"]
    df_stat["sales_max"] = df_stat["mean"] + cut_stdev * df_stat["std"]

    df_compare = df.join(df_stat[["sales_min", "sales_max"]], on="vegetable")

    df["is_outlier"] = (df["sales"] < df_compare["sales_min"]) | (
        df["sales"] > df_compare["sales_max"]
    )

    return df
