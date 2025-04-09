import pandas as pd
import pytest
from sklearn.metrics import r2_score

import main


def test_model_prev_month():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
        },
        "start_test": "2023-07-01",
        "model": "PrevMonthSale",
    }

    df_pred = main.make_predictions(config)

    df_expected = pd.read_csv("data/raw/prediction_prev_month.csv")

    df_pred = sort_dataframe(df_pred)
    df_expected = sort_dataframe(df_expected)

    pd.testing.assert_frame_equal(df_expected, df_pred)


def test_model_same_month_last_year():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
        },
        "start_test": "2023-07-01",
        "model": "SameMonthLastYearSales",
    }

    df_pred = main.make_predictions(config)

    df_expected = pd.read_csv("data/raw/prediction_same_month_last_year.csv")

    df_pred = sort_dataframe(df_pred)
    df_expected = sort_dataframe(df_expected)

    pd.testing.assert_frame_equal(df_expected, df_pred)


def test_autoregressive_model():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
        },
        "start_test": "2023-07-01",
        "model": "Ridge",
        "features": ["past_sales", "marketing"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])
    assert r2 == pytest.approx(0.8019, rel=1e-3)


def test_marketing_model():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
        },
        "start_test": "2023-07-01",
        "model": "Ridge",
        "features": ["past_sales", "marketing"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])
    assert r2 == pytest.approx(0.8019, rel=1e-3)


def test_price_model():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
            "price": "data/raw/price.csv",
        },
        "start_test": "2023-07-01",
        "model": "Ridge",
        "features": ["past_sales", "marketing", "price"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])

    assert r2 == pytest.approx(0.8446, rel=1e-3)


def test_stock_model():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
            "price": "data/raw/price.csv",
            "stock": "data/raw/stock.csv",
        },
        "start_test": "2023-07-01",
        "model": "Ridge",
        "features": ["past_sales", "marketing", "price", "stock"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])

    assert r2 == pytest.approx(0.8446, rel=1e-3)


def test_model_with_objectives():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
            "price": "data/raw/price.csv",
            "stock": "data/raw/stock.csv",
            "objectives": "data/raw/objectives.csv",
        },
        "start_test": "2023-07-01",
        "model": "Ridge",
        "features": ["past_sales", "marketing", "price", "stock", "objectives"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])

    assert r2 == pytest.approx(0.8446, rel=1e-3)


def test_custom_model():
    config = {
        "data": {
            "sales": "data/raw/sales.csv",
            "marketing": "data/raw/marketing.csv",
            "price": "data/raw/price.csv",
            "stock": "data/raw/stock.csv",
            "objectives": "data/raw/objectives.csv",
        },
        "start_test": "2023-07-01",
        "model": "CustomModel",
        "features": ["past_sales", "marketing", "price", "stock", "objectives"],
    }

    df_pred = main.make_predictions(config)

    df_true = pd.read_csv("data/raw/sales_to_predict.csv")

    df_pred = sort_dataframe(df_pred)
    df_true = sort_dataframe(df_true)

    r2 = r2_score(df_true["sales"], df_pred["prediction"])

    assert r2 == pytest.approx(0.8446, rel=1e-3)


def sort_dataframe(df):
    return (
        df.astype({"dates": str})
        .sort_values(["item_id", "dates"], ignore_index=True)
        .reset_index(drop=True)
    )
