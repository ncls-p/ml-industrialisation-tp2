import requests

URL = "http://127.0.0.1:8000/"


def test_idempotency():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    for _ in range(2):
        requests.post(
            URL + "post_sales",
            json=[{"year_week": 202001, "vegetable": "tomato", "sales": 100}],
        )

    response = requests.get(
        URL + "get_raw_sales",
    )

    assert response.status_code == 200

    assert response.json() == [
        {"year_week": 202001, "vegetable": "tomato", "sales": 100}
    ]


def test_get_monthly_sales():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    requests.post(
        URL + "post_sales",
        json=[{"year_week": 202004, "vegetable": "tomato", "sales": 70}],
    )

    response = requests.get(
        URL + "get_monthly_sales",
    )

    assert response.status_code == 200

    assert response.json() == [
        {"year_month": 202001, "vegetable": "tomato", "sales": 50},
        {"year_month": 202002, "vegetable": "tomato", "sales": 20},
    ]


def test_tag_outlier__any():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    for i in range(1, 28):
        requests.post(
            URL + "post_sales",
            json=[{"year_week": 202000 + i, "vegetable": "tomato", "sales": 70}],
        )

    requests.post(
        URL + "post_sales",
        json=[{"year_week": 202050, "vegetable": "tomato", "sales": 100_000}],
    )

    response = requests.get(
        URL + "get_monthly_sales",
    )

    assert response.status_code == 200

    assert response.json() == [
        {"year_month": 202001, "vegetable": "tomato", "sales": 260},
        {"year_month": 202002, "vegetable": "tomato", "sales": 290},
        {"year_month": 202003, "vegetable": "tomato", "sales": 310},
        {"year_month": 202004, "vegetable": "tomato", "sales": 300},
        {"year_month": 202005, "vegetable": "tomato", "sales": 310},
        {"year_month": 202006, "vegetable": "tomato", "sales": 300},
        {"year_month": 202007, "vegetable": "tomato", "sales": 120},
        {"year_month": 202012, "vegetable": "tomato", "sales": 100_000},
    ]

    response = requests.get(URL + "get_monthly_sales", params={"remove_outliers": True})

    assert response.status_code == 200

    assert response.json() == [
        {"year_month": 202001, "vegetable": "tomato", "sales": 260},
        {"year_month": 202002, "vegetable": "tomato", "sales": 290},
        {"year_month": 202003, "vegetable": "tomato", "sales": 310},
        {"year_month": 202004, "vegetable": "tomato", "sales": 300},
        {"year_month": 202005, "vegetable": "tomato", "sales": 310},
        {"year_month": 202006, "vegetable": "tomato", "sales": 300},
        {"year_month": 202007, "vegetable": "tomato", "sales": 120},
    ]


def test_a_baddly_written_row():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    requests.post(
        URL + "post_sales",
        json=[
            {"year_week": 202003, "vegetable": "Tomatto", "sales": 70},
            {"year_week": 202003, "vegetable": "Tomate", "sales": 70},
            {"year_week": 202004, "vegetable": "tomato", "sales": 70},
        ],
    )

    response = requests.get(
        URL + "get_monthly_sales",
    )

    assert response.status_code == 200

    assert response.json() == [
        {"year_month": 202001, "vegetable": "tomato", "sales": 190},
        {"year_month": 202002, "vegetable": "tomato", "sales": 20},
    ]


def test_one_bad_row():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    requests.post(
        URL + "post_sales",
        json=[
            {"year_week": 202003, "vegetable": "Tomatto", "sales": 70},
            {"year_week": 202003, "vegetable": None, "sales": 70},
            {"year_week": 202004, "vegetable": "tomato", "sales": 70},
        ],
    )

    response = requests.get(
        URL + "get_monthly_sales",
    )

    assert response.status_code == 200

    assert response.json() == [
        {"year_month": 202001, "vegetable": "tomato", "sales": 120},
        {"year_month": 202002, "vegetable": "tomato", "sales": 20},
    ]


def test_tag_outlier__compute_stat_by_vegetable():
    response_init_db = requests.get(URL + "init_db")
    assert response_init_db.status_code == 200

    for i in range(1, 15):
        requests.post(
            URL + "post_sales",
            json=[
                {"year_week": 202000 + i, "vegetable": "tomato", "sales": 70},
                {"year_week": 202000 + i, "vegetable": "pear", "sales": 70},
            ],
        )

    requests.post(
        URL + "post_sales",
        json=[{"year_week": 202050, "vegetable": "tomato", "sales": 100_000}],
    )

    response = requests.get(URL + "get_monthly_sales", params={"remove_outliers": True})

    assert response.status_code == 200

    expected_reply = [
        {"year_month": 202001, "vegetable": "pear", "sales": 260},
        {"year_month": 202001, "vegetable": "tomato", "sales": 260},
        {"year_month": 202002, "vegetable": "pear", "sales": 290},
        {"year_month": 202002, "vegetable": "tomato", "sales": 290},
        {"year_month": 202003, "vegetable": "pear", "sales": 310},
        {"year_month": 202003, "vegetable": "tomato", "sales": 310},
        {"year_month": 202004, "vegetable": "pear", "sales": 120},
        {"year_month": 202004, "vegetable": "tomato", "sales": 120},
        {"year_month": 202012, "vegetable": "tomato", "sales": 100_000},
    ]

    assert (
        sorted(response.json(), key=lambda r: (r["year_month"], r["vegetable"]))
        == expected_reply
    )
