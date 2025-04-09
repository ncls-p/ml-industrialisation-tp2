import os

import pandas as pd
from flask import Flask, jsonify, request

from api.data import clean_vegetable_name, compute_monthly_sales, tag_outliers

PATH_CSV = "data/raw/db.csv"


def create_app(config=None):
    config = config or {}
    app = Flask(__name__)

    if "CSV_PATH" not in config:
        config["CSV_PATH"] = PATH_CSV

    app.config.update(config)

    @app.route("/post_sales", methods=["POST"])
    def post_sales():
        data = request.json
        df_new = pd.DataFrame(data)

        if (
            os.path.isfile(app.config["CSV_PATH"])
            and os.path.getsize(app.config["CSV_PATH"]) > 0
        ):
            df = pd.read_csv(app.config["CSV_PATH"])
            df = pd.concat([df, df_new])
        else:
            df = df_new

        df = df.drop_duplicates(subset=["year_week", "vegetable"])

        df.to_csv(app.config["CSV_PATH"], index=False)

        return jsonify({"status": "success"}), 200

    @app.route("/get_raw_sales", methods=["GET"])
    def get_raw_sales():
        df = pd.read_csv(app.config["CSV_PATH"])
        return jsonify(df.to_dict("records")), 200

    @app.route("/get_monthly_sales", methods=["GET"])
    def get_monthly_sales():
        df = pd.read_csv(app.config["CSV_PATH"])
        df = clean_vegetable_name(df)

        if request.args.get("remove_outliers"):
            df = tag_outliers(df)
            df = df[~df["is_outlier"]].drop(columns=["is_outlier"])

        df_month = compute_monthly_sales(df)

        return jsonify(df_month.to_dict("records")), 200  # type: ignore

    @app.route("/init_db", methods=["GET"])
    def init_db():
        if os.path.isfile(app.config["CSV_PATH"]):
            os.remove(config["CSV_PATH"])
        return jsonify({"status": "success"}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=8000, debug=True)
