# TD2 Simplification Plan

## Objective

Simplify `src/main.py` to **strictly** implement the requirements of TD2 from the README, while **ensuring all tests in `tests/test_model.py` pass without modification**.

---

## Key Requirements

- **Support exactly four models:**

  - `PrevMonthSale`
  - `SameMonthLastYearSales`
  - `Ridge` (autoregressive with optional features)
  - `CustomModel` (parametric, optimized with scipy)

- **Support all external data sources:**

  - Marketing
  - Price
  - Stock
  - Objectives

- **Gracefully handle missing external data** (empty DataFrames fallback).

- **Configurable feature inclusion** for Ridge and CustomModel via `config["features"]`.

- **Maintain the API:** `make_predictions(config: dict) -> pd.DataFrame`

---

## Pipeline Overview

```mermaid
flowchart TD
    A[Config dict] --> B[load_data()]
    B --> C[feature_engineering()]
    C --> D{Model selection}
    D -->|PrevMonthSale| E[prev_month_sale_model()]
    D -->|SameMonthLastYearSales| F[same_month_last_year_model()]
    D -->|Ridge| G[ridge_autoregressive_model()]
    D -->|CustomModel| H[custom_parametric_model()]
    E --> I[adjust_for_objectives()]
    F --> I
    G --> I
    H --> I
    I --> J[Filter test period]
    J --> K[Return predictions]
```

---

## Planned Simplifications

### 1. Data Loading

- Keep `load_data(config)` with optional CSV loading.
- Simplify error handling where possible.
- Always return empty DataFrames if files missing.

### 2. Feature Engineering

- Compute:
  - `lag_12` (sales same month last year)
  - `avg_12` (average sales over last 12 months)
  - `feat_c` (lag_12 \* quarter-over-quarter growth)
- Merge external data sources.
- Compute fiscal year, cumulative sales.
- Flag stockout months for exclusion and capping.

### 3. Models

- **PrevMonthSale:** predict previous month sales.
- **SameMonthLastYearSales:** predict sales same month last year.
- **Ridge:**
  - Use features from `config["features"]`.
  - Train on data before `start_test`.
  - Exclude stockout months if stock feature used.
  - Cap predictions by stock if stock feature used.
- **CustomModel:**
  - Parametric formula matching data generation.
  - Optimize parameters with scipy.
  - Cap predictions by stock.

### 4. Objective Adjustment

- Adjust June/July predictions based on fiscal year objectives.

### 5. Main Function

- `make_predictions(config)`:
  - Load data.
  - Feature engineering.
  - Select and run model.
  - Adjust for objectives.
  - Return predictions for test period.

---

## What will be removed

- Any models or options **not** covered by TD2 or tests.
- Overly defensive programming.
- Unused or redundant code.
- Excessive flexibility beyond test requirements.

---

## Expected Outcome

- A **clean, minimal** implementation.
- Fully compliant with TD2 instructions.
- Passes all existing tests in `tests/test_model.py`.
- Easier to maintain and understand.
