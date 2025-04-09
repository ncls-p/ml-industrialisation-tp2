import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def load_data(config):
    df_sales = pd.read_csv(config["data"]["sales"])
    df_sales = df_sales.rename(columns={"sales": "sales_target"})
    df_sales["dates"] = pd.to_datetime(df_sales["dates"])

    def parse_dates(df_ext):
        if "dates" in df_ext.columns:
            df_ext["dates"] = pd.to_datetime(df_ext["dates"])

    def safe_read(path, keys):
        if path is None:
            return pd.DataFrame(columns=keys)
        try:
            df_tmp = pd.read_csv(path)
            if "date" in df_tmp.columns and "dates" not in df_tmp.columns:
                df_tmp.rename(columns={"date": "dates"}, inplace=True)
            parse_dates(df_tmp)
            return df_tmp
        except (KeyError, FileNotFoundError, TypeError, ValueError):
            return pd.DataFrame(columns=keys)

    df_marketing = safe_read(
        config["data"].get("marketing"), ["item_id", "dates", "marketing_spend"]
    )
    df_price = safe_read(config["data"].get("price"), ["item_id", "dates", "price"])
    df_stock = safe_read(config["data"].get("stock"), ["item_id", "dates", "stock"])
    df_objectives = safe_read(
        config["data"].get("objectives"), ["item_id", "objective"]
    )

    for df_ext in [df_marketing, df_price, df_stock, df_objectives]:
        if "sales" in df_ext.columns:
            df_ext.drop(columns=["sales"], inplace=True)

    external_data = {
        "marketing": df_marketing,
        "price": df_price,
        "stock": df_stock,
        "objectives": df_objectives,
    }

    return df_sales, external_data


def feature_engineering(df_sales, external_data):
    df = df_sales.copy()
    df = df.sort_values(["item_id", "dates"])

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
    df["qoq_growth"] = df["recent_sales"] / df["past_sales"].replace(0, 1)
    df["qoq_growth"] = df["qoq_growth"].fillna(1)
    df["feat_c"] = df["lag_12"] * df["qoq_growth"]

    if not external_data["marketing"].empty:
        df = df.merge(external_data["marketing"], on=["item_id", "dates"], how="left")
        df["marketing_spend"] = df["marketing_spend"].fillna(0)

    if not external_data["price"].empty:
        df = df.merge(external_data["price"], on=["item_id", "dates"], how="left")
        df = df.sort_values(["item_id", "dates"])
        df["future_price"] = (
            df.groupby("item_id")["price"].shift(-1).fillna(df["price"])
        )
        df["price_change"] = (df["future_price"] - df["price"]) / df["price"].replace(
            0, 1
        )
        df["price_change"] = df["price_change"].fillna(0)

    if not external_data["stock"].empty:
        df = df.merge(external_data["stock"], on=["item_id", "dates"], how="left")
        df["stock_end_month"] = df["stock"].fillna(np.inf)
        df["stockout"] = df["stock_end_month"] == 0
        df["stockout_prev"] = df.groupby("item_id")["stockout"].shift(1).fillna(False)
        df["exclude"] = df["stockout"] | df["stockout_prev"]
    else:
        df["stock_end_month"] = np.inf
        df["stockout"] = False
        df["stockout_prev"] = False
        df["exclude"] = False

    if not external_data["objectives"].empty:
        df = df.merge(external_data["objectives"], on=["item_id"], how="left")
        df["objective"] = df["objective"].fillna(0)
        df["target_obj"] = df["objective"]
    else:
        df["objective"] = 0
        df["target_obj"] = 0

    df["month"] = df["dates"].dt.month
    df["year"] = df["dates"].dt.year
    df["fiscal_year"] = df["year"].copy()
    df.loc[df["month"] > 6, "fiscal_year"] += 1
    df["fiscal_year_month"] = (
        df["fiscal_year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.sort_values(["item_id", "fiscal_year", "month"])
    df["cum_sales_fiscal"] = df.groupby(["item_id", "fiscal_year"])[
        "sales_target"
    ].cumsum()

    return df


def same_month_last_year_model(df):
    df["prediction"] = df["lag_12"].fillna(0)
    return df


def prev_month_sale_model(df):
    df["prediction"] = df.groupby("item_id")["sales_target"].shift(1).fillna(0)
    return df


def ridge_autoregressive_model(df, config):
    feature_map = {
        "past_sales": ["lag_12", "avg_12", "feat_c"],
        "marketing": ["marketing_spend"],
        "price": ["price_change"],
        "stock": ["stock_end_month", "stockout", "stockout_prev"],
        "objectives": ["target_obj"],
    }

    selected_features = []
    for feat in config.get("features", ["past_sales"]):
        if feat in feature_map:
            selected_features.extend(feature_map[feat])

    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    train_mask = df["dates"] < pd.to_datetime(config["start_test"])
    if "stock" in config.get("features", []):
        train_mask = train_mask & ~df["exclude"]

    X_train = df.loc[train_mask, selected_features].fillna(0)
    y_train = df.loc[train_mask, "sales_target"]

    y_train = pd.to_numeric(y_train, errors="coerce")
    valid_idx = ~y_train.isna()
    X_train = X_train.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

    model = Ridge(alpha=450.0)
    model.fit(X_train, y_train)

    X_all = df[selected_features].fillna(0)
    df["prediction"] = model.predict(X_all)

    if "stock" in config.get("features", []) and "stock_end_month" in df.columns:
        df["prediction"] = np.minimum(
            df["stock_end_month"].fillna(np.inf), df["prediction"]
        )

    return df


def custom_parametric_model(df, config):
    train_mask = df["dates"] < pd.to_datetime(config["start_test"])
    df_train = df.loc[train_mask].copy()

    def model_func(params, df_sub):
        a, b, c, d, e = params
        lag_12 = df_sub["lag_12"].fillna(0)
        avg_12 = df_sub["avg_12"].fillna(0)
        feat_c = df_sub["feat_c"].fillna(0)
        marketing = (
            df_sub["marketing_spend"].fillna(0)
            if "marketing_spend" in df_sub.columns
            else 0
        )
        price_change = (
            df_sub["price_change"].fillna(0) if "price_change" in df_sub.columns else 0
        )
        pred = a * lag_12 + b * avg_12 + c * feat_c
        pred *= (1 + d * marketing) * (1 + e * price_change)
        if "stock_end_month" in df_sub.columns:
            stock = df_sub["stock_end_month"].fillna(np.inf)
            pred = np.minimum(stock, pred)
        return pred

    def loss(params):
        pred = model_func(params, df_train)
        y_true = df_train["sales_target"].fillna(0)
        return np.mean((y_true - pred) ** 2)

    initial_params = [0.7, 0.2, 0.1, 0.05, -0.1]
    res = minimize(loss, x0=initial_params, method="L-BFGS-B")
    opt_params = res.x

    df["prediction"] = model_func(opt_params, df)

    return df


def adjust_for_objectives(df):
    df = df.copy()
    if "objective" in df.columns and df["objective"].max() > 0:
        for item_id in df["item_id"].unique():
            item_mask = df["item_id"] == item_id
            for year in df.loc[item_mask, "fiscal_year"].unique():
                year_mask = df["fiscal_year"] == year
                june_mask = (item_mask) & (year_mask) & (df["month"] == 6)
                july_mask = (item_mask) & (year_mask) & (df["month"] == 7)
                if not june_mask.any() or not july_mask.any():
                    continue
                objective = df.loc[june_mask, "objective"].iloc[0]
                if objective <= 0:
                    continue
                may_mask = (item_mask) & (year_mask) & (df["month"] == 5)
                if may_mask.any():
                    cum_sales = df.loc[may_mask, "cum_sales_fiscal"].iloc[0]
                    percent_to_objective = cum_sales / objective
                    if 0.8 <= percent_to_objective < 0.99:
                        df.loc[june_mask, "prediction"] *= 1.2
                        df.loc[july_mask, "prediction"] *= 0.8
                    elif percent_to_objective < 0.8:
                        pass
    return df


def make_predictions(config):
    df_sales = pd.read_csv(config["data"]["sales"])
    df_sales["dates"] = pd.to_datetime(df_sales["dates"])

    model_name = config.get("model", "PrevMonthSale")

    if model_name == "PrevMonthSale":
        df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(1)

        df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(
            drop=True
        )

        return df_sales[["dates", "item_id", "prediction"]]

    elif model_name == "SameMonthLastYearSales":
        df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(12)

        df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(
            drop=True
        )

        return df_sales[["dates", "item_id", "prediction"]]

    elif model_name == "Ridge":
        # Load external data if specified, else empty DataFrames
        external_data = {}
        for key in ["marketing", "price", "stock", "objectives"]:
            path = config.get("data", {}).get(key, None)
            if path:
                external_data[key] = pd.read_csv(path)
                if "dates" in external_data[key].columns:
                    external_data[key]["dates"] = pd.to_datetime(
                        external_data[key]["dates"]
                    )
            else:
                external_data[key] = pd.DataFrame()

        # Prepare sales_target column
        df_sales["sales_target"] = df_sales["sales"]

        # Feature engineering
        df_features = feature_engineering(df_sales, external_data)

        # Model training and prediction
        df_pred = ridge_autoregressive_model(df_features, config)

        df_pred = df_pred[df_pred["dates"] >= config["start_test"]].reset_index(
            drop=True
        )

        return df_pred[["dates", "item_id", "prediction"]]

    else:
        raise ValueError(f"Unknown model: {model_name}")
