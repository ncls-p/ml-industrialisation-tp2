from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

DB_URI = "sqlite:///your_database.db"

DB = SQLAlchemy()


def create_app(config):
    app = Flask(__name__)

    if "SQLALCHEMY_DATABASE_URI" not in config:
        config["SQLALCHEMY_DATABASE_URI"] = DB_URI

    app.config.update(config)

    DB.init_app(app)

    with app.app_context():
        from api.models import SaleWeeklyRaw

        DB.create_all()

    @app.route("/post_sales", methods=["POST"])
    def post_sales():
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        for row in data:
            try:
                new_entry = SaleWeeklyRaw(  # type: ignore
                    year_week=row.get("year_week"),  # type: ignore
                    vegetable=row.get("vegetable"),  # type: ignore
                    sales=row.get("sales"),  # type: ignore
                )
                DB.session.add(new_entry)
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400

        DB.session.commit()

        return jsonify({"status": "success"}), 200

    @app.route("/get_raw_data", methods=["GET"])
    def get_raw_data():
        entries = DB.session.query(SaleWeeklyRaw).filter_by(
            year_week=202001,
            vegetable="babar",
        )

        return jsonify([row.json() for row in entries]), 200

    return app
