from api.app_sql import DB


class SaleWeeklyRaw(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    year_week = DB.Column(DB.Integer, nullable=False)
    vegetable = DB.Column(DB.String(80), nullable=False)
    sales = DB.Column(DB.Float, nullable=False)

    def json(self):
        return {
            "year_week": self.year_week,
            "vegetable": self.vegetable,
            "sales": self.sales,
        }
