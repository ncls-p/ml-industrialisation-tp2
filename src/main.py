import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def load_external_data(config):
    data_dict = {}
    expected_sources = ["marketing", "price", "stock", "objectives"]
    for src in expected_sources:
        path = config.get("data", {}).get(src)
        if not path and src == "marketing":
            path = "data/raw/marketing.csv"
        try:
            if path:
                df = pd.read_csv(path)
                if "date" in df.columns and "dates" not in df.columns:
                    df.rename(columns={"date": "dates"}, inplace=True)
                if "dates" in df.columns:
                    df["dates"] = pd.to_datetime(df["dates"])
                data_dict[src] = df
            else:
                data_dict[src] = pd.DataFrame()
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
            data_dict[src] = pd.DataFrame()
    return data_dict


def build_features(df_sales, ext_data):
    df = df_sales.copy().sort_values(["item_id", "dates"])
    df["lag_12"] = df.groupby("item_id")["sales_target"].shift(12)
    df["avg_12"] = df.groupby("item_id")["sales_target"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )
    df["recent_sales"] = df.groupby("item_id")["sales_target"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["past_sales"] = df.groupby("item_id")["sales_target"].transform(
        lambda x: x.shift(13).rolling(3, min_periods=1).sum()
    )
    df["qoq_growth"] = (df["recent_sales"] / df["past_sales"].replace(0, 1)).fillna(1)
    df["feat_c"] = df["lag_12"] * df["qoq_growth"]

    def safe_merge(base_df, key, cols):
        info = ext_data.get(key)
        if info is not None and not info.empty:
            return base_df.merge(info, on=cols, how="left")
        return base_df

    df = safe_merge(df, "marketing", ["item_id", "dates"])
    df["marketing_spend"] = df.get(
        "marketing_spend", pd.Series(0, index=df.index)
    ).fillna(0)

    df = safe_merge(df, "price", ["item_id", "dates"])
    if "price" in df.columns:
        df = df.sort_values(["item_id", "dates"])
        df["future_price"] = (
            df.groupby("item_id")["price"].shift(-1).fillna(df["price"])
        )
        df["price_change"] = (df["future_price"] - df["price"]) / df["price"].replace(
            0, 1
        )
        df["price_effect"] = df["price_change"].fillna(0)
        df["price"] = df["price"].ffill().bfill()
    else:
        df["price_change"] = df["price_effect"] = 0
        df["price"] = 0

    df = safe_merge(df, "stock", ["item_id", "dates"])
    if "stock" in df.columns:
        df["stock_end_month"] = df["stock"].fillna(np.inf)
        df["stockout"] = df["stock_end_month"] == 0
        df["stockout_prev"] = df.groupby("item_id")["stockout"].shift(1).fillna(False)
        df["exclude"] = df["stockout"] | df["stockout_prev"]
    else:
        df["stock_end_month"] = np.inf
        df["stockout"] = df["stockout_prev"] = df["exclude"] = False

    df = safe_merge(df, "objectives", ["item_id"])
    df["objective"] = df.get("objective", pd.Series(0, index=df.index)).fillna(0)
    df["target_obj"] = df["objective"]

    df["month"] = df["dates"].dt.month
    df["year"] = df["dates"].dt.year
    df["fiscal_year"] = df["year"]
    df.loc[df["month"] > 6, "fiscal_year"] += 1
    df["fiscal_year_month"] = (
        df["fiscal_year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.sort_values(["item_id", "fiscal_year", "month"])
    df["cum_sales_fiscal"] = df.groupby(["item_id", "fiscal_year"])[
        "sales_target"
    ].cumsum()

    return df


def baseline_model(df, mode):
    df = df.copy()
    if mode == "SameMonthLastYearSales":
        if "lag_12" not in df.columns:
            df = df.sort_values(["item_id", "dates"])
            df["lag_12"] = df.groupby("item_id")["sales_target"].shift(12)
        df["prediction"] = df["lag_12"].fillna(0)
    elif mode == "PrevMonthSale":
        df["prediction"] = df.groupby("item_id")["sales_target"].shift(1).fillna(0)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
    return df


def ridge_autoregressive_model(df, config):
    feature_map = {
        "past_sales": ["lag_12", "avg_12", "feat_c"],
        "marketing": ["marketing_spend"],
        "price": ["price", "price_effect"],
        "stock": ["stock_end_month", "stockout", "stockout_prev"],
        "objectives": ["target_obj"],
    }
    feat_list = []
    for group in config.get("features", ["past_sales"]):
        feat_list.extend(feature_map.get(group, []))

    for feat in feat_list:
        if feat not in df.columns:
            df[feat] = 0

    mask_train = df["dates"] < pd.to_datetime(config["start_test"])
    if "stock" in config.get("features", []):
        mask_train &= ~df.get("exclude", False)

    X_train = df.loc[mask_train, feat_list].fillna(0)
    y_train = df.loc[mask_train, "sales_target"]
    valid_idx = ~pd.to_numeric(y_train, errors="coerce").isna()
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    alpha = 20 if "price" in config.get("features", []) else 450
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    df["prediction"] = model.predict(df[feat_list].fillna(0))

    if "stock" in config.get("features", []) and "stock_end_month" in df.columns:
        df["prediction"] = np.minimum(
            df["stock_end_month"].fillna(np.inf), df["prediction"]
        )

    return df


def custom_parametric_model(df, config):
    mask_train = df["dates"] < pd.to_datetime(config["start_test"])
    df_train = df.loc[mask_train].copy()

    def model_func(params, d):
        a, b, c, d_coef, e_coef = params
        base = (
            a * d["lag_12"].fillna(0)
            + b * d["avg_12"].fillna(0)
            + c * d["feat_c"].fillna(0)
        )
        marketing_multiplier = 1 + d_coef * d.get("marketing_spend", 0).fillna(0)
        price_multiplier = 1 + e_coef * d.get("price_change", 0).fillna(0)
        pred = base * marketing_multiplier * price_multiplier

        if "stock_end_month" in d.columns:
            pred = np.minimum(d["stock_end_month"].fillna(np.inf), pred)

        return pred

    def loss(params):
        preds = model_func(params, df_train)
        return np.mean((df_train["sales_target"].fillna(0) - preds) ** 2)

    initial_guess = [0.7, 0.2, 0.1, 0.05, -0.1]
    res = minimize(loss, initial_guess, method="L-BFGS-B")
    params_opt = res.x

    df["prediction"] = model_func(params_opt, df)
    return df


def adjust_for_objectives(df):
    df = df.copy()
    if "objective" not in df.columns or df["objective"].max() <= 0:
        return df
    for item in df["item_id"].unique():
        mask_item = df["item_id"] == item
        fiscal_years = df.loc[mask_item, "fiscal_year"].unique()
        for fy in fiscal_years:
            mask_fy = (mask_item) & (df["fiscal_year"] == fy)
            mask_june = mask_fy & (df["month"] == 6)
            mask_july = mask_fy & (df["month"] == 7)
            if not (mask_june.any() and mask_july.any()):
                continue
            obj = df.loc[mask_june, "objective"].iloc[0]
            if obj <= 0:
                continue
            mask_may = mask_fy & (df["month"] == 5)
            if not mask_may.any():
                continue
            cum_sales = df.loc[mask_may, "cum_sales_fiscal"].iloc[0]
            pct_achieved = cum_sales / obj
            if 0.8 <= pct_achieved < 0.99:
                df.loc[mask_june, "prediction"] *= 1.2
                df.loc[mask_july, "prediction"] *= 0.8
    return df


def make_predictions(config):
    df_sales = pd.read_csv(config["data"]["sales"])
    df_sales["dates"] = pd.to_datetime(df_sales["dates"])
    df_sales["sales_target"] = df_sales["sales"]

    model_type = config.get("model", "PrevMonthSale")

    if model_type in ("PrevMonthSale", "SameMonthLastYearSales"):
        df_sales = baseline_model(df_sales, model_type)
        output = df_sales[df_sales["dates"] >= pd.to_datetime(config["start_test"])]

    else:
        ext_data = load_external_data(config)
        df_feat = build_features(df_sales, ext_data)
        if model_type == "Ridge":
            df_pred = ridge_autoregressive_model(df_feat, config)
        elif model_type == "CustomModel":
            df_pred = custom_parametric_model(df_feat, config)
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        df_pred = adjust_for_objectives(df_pred)
        output = df_pred[df_pred["dates"] >= pd.to_datetime(config["start_test"])]

    return output.reset_index(drop=True)[["dates", "item_id", "prediction"]]
