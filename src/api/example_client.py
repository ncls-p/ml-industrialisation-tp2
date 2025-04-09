import requests

response = requests.post(
    "http://127.0.0.1:8000/post_sales",
    json=[{"year_week": 202001, "vegetable": "tomato", "sales": 100}],
)

print(response.status_code)
