# Real Estate House Price Prediction using Gradient Boosting & Regression

A machine learning project that predicts residential real estate valuation and property market prices using property dimensions, spatial locations, amenities, and neighborhood characteristics.

## Project Overview

Accurate property valuation is critical for buyers, sellers, mortgage lenders, and real estate investment platforms. Traditional appraisal methods rely on manual human inspection and isolated recent sales comparables, which can be subjective, slow, and prone to market lag.

This project implements an automated, data-driven machine learning valuation model utilizing **Gradient Boosting Regression** and **Random Forest Regressors** to estimate fair market value for residential properties based on key architectural, temporal, and spatial factors:

* Built-up area (square footage)
* Room configuration (bedrooms and bathrooms)
* Neighborhood classification (`Urban`, `Suburban`, `Rural`)
* Distance to central business district (CBD)
* Age of construction and structural depreciation
* Garage and parking capacity
* Private garden / backyard amenities
* Local educational infrastructure (school district rating)

The project includes complete exploratory data analysis (EDA), multi-variable correlation checks, automated categorical encoding, feature scaling, multi-model evaluation ($R^2$, RMSE, MAE), residual error analysis, feature importance interpretation, and an interactive property valuation tool with live UI sliders.

## Business Problem

The objective is to answer:

> "Given the physical attributes, location, age, and surrounding neighborhood infrastructure of a residential property, what is its estimated market valuation in Lakhs (₹)?"

The model delivers:

* A precise estimated property price (in Lakhs ₹)
* A calculated 90% confidence valuation interval
* Breakdown of value contributors (e.g., premium added by urban location or penalty from distance/age)

In a commercial real estate or fintech environment, this model powers automated valuation models (AVMs), mortgage collateral risk assessment, and algorithmic real estate listing pricing.

## Dataset

The project uses `house_price_data.csv`, containing 1,500 residential property records.

### Features

| Feature | Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| `property_id` | Identifier | Unique listing identifier | `PROP-1001` to `PROP-2500` |
| `square_feet` | Numerical | Built-up living area | 650 – 4,200 sq ft |
| `bedrooms` | Numerical | Number of bedrooms | 1 – 5 |
| `bathrooms` | Numerical | Number of bathrooms | 1 – 4 |
| `location_type` | Categorical | Neighborhood density | `Urban`, `Suburban`, `Rural` |
| `property_age_years`| Numerical | Age of the property since construction | 0 – 35 years |
| `garage_spaces` | Numerical | Covered vehicle parking spaces | 0 – 3 |
| `has_garden` | Categorical | Presence of private garden/lawn | `Yes`, `No` |
| `distance_to_city_center_km` | Numerical | Distance to urban center / CBD | 1.0 – 32.0 km |
| `school_rating` | Numerical | Quality rating of nearest school zone | 2.0 – 10.0 |

### Target Variable

`price_in_lakhs`

* Continuous numerical variable representing estimated market price in Lakhs (1 Lakh = ₹100,000).

## Machine Learning Approach

The workflow follows a rigorous supervised regression pipeline:

```text
house_price_data.csv
        ↓
Exploratory Data Analysis (EDA) & Outlier Check
        ↓
Preprocessing & One-Hot Encoding (ColumnTransformer)
        ↓
Train/Test Split (80% Train, 20% Test)
        ↓
Feature Scaling (StandardScaler)
        ↓
Model Training (Gradient Boosting Regressor)
        ↓
Model Comparison (vs Random Forest & Ridge Regression)
        ↓
Evaluation (R², RMSE, MAE, Residual Plots)
        ↓
Feature Importance Analysis
        ↓
Interactive Valuation Estimator
```

### Why Gradient Boosting Regressor?

Gradient Boosting was chosen because:

1. **Non-linear feature interactions**: Property pricing is non-linear (e.g., square footage matters much more in Urban locations than Rural locations).
2. **Robustness to scale differences**: Handles diverse feature scales (square feet in thousands vs bathrooms in single digits) seamlessly.
3. **Sequential error correction**: Each subsequent weak tree optimizes against the pseudo-residuals of earlier trees, achieving higher $R^2$ accuracy than standard linear regressions.
4. **Generalization**: Controlled learning rates and tree depths prevent overfitting on small local property clusters.

## Model Evaluation

The model is evaluated on held-out test properties using standard continuous regression metrics:

### $R^2$ Score (Coefficient of Determination)
Measures the proportion of variance in property prices explained by the features (typically > 0.90 for this architecture).

### RMSE (Root Mean Squared Error)
Penalizes large valuation discrepancies heavily, reflecting real estate appraisal risk.

### MAE (Mean Absolute Error)
Measures the average magnitude of absolute dollar error across predictions in Lakhs.

### Residual Analysis
Examines whether valuation errors are normally distributed around zero with constant variance (homoscedasticity).

## Key Price Drivers (Feature Importance)

Analysis reveals the primary determinants of residential value:

1. **Square Footage (`square_feet`)**: The single largest driver of total property value.
2. **Location Classification (`location_type`)**: Urban properties carry a significant per-square-foot premium over Suburban and Rural equivalents.
3. **Distance to City Center (`distance_to_city_center_km`)**: Strong negative coefficient — each additional kilometer away from the CBD reduces property desirability.
4. **Property Age (`property_age_years`)**: Natural structural depreciation and outdated interiors negatively impact pricing.
5. **School Rating & Amenities**: High school ratings and private gardens provide clear positive value increments.

## Interactive Valuation Demo

The notebook provides a live slider-based calculator powered by `ipywidgets`.

Users can configure:
* Living area (700 to 4,000 sq ft)
* Bedrooms (1 to 5) and Bathrooms (1 to 4)
* Neighborhood (`Urban`, `Suburban`, `Rural`)
* Property age (0 to 30 years)
* Covered parking spaces (0 to 3)
* Distance to center and school rating

The tool outputs an immediate estimated price in Lakhs, along with a ±5% valuation range.

## Technologies Used

* **Python 3.10+**
* **Pandas & NumPy** — Structured data processing and mathematical routines
* **Scikit-learn** — Gradient Boosting, Random Forest, Ridge Regression, Preprocessing pipelines
* **Matplotlib & Seaborn** — Regression plots, residual diagnostics, correlation matrices
* **IPyWidgets** — Interactive UI sliders for Colab and Jupyter Notebook
* **Jupyter Notebook / Google Colab** — Interactive model experimentation

## Project Structure

```text
house-price-prediction-model--main/
└── house-price-prediction-model--main/
    ├── LICENSE
    ├── README.md
    ├── house_price_data.csv
    └── house price prediction model .ipynb
```

## How to Run

### Option 1: Google Colab
1. Upload the notebook `house price prediction model .ipynb` to Google Colab.
2. Upload `house_price_data.csv` to the session files.
3. Run `Runtime -> Run all`.
4. Test hypothetical properties using the interactive calculator at the bottom.

### Option 2: Local Jupyter Notebook
```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets
jupyter notebook "house price prediction model .ipynb"
```

## Limitations

* Pricing does not account for specific micro-locational features (e.g. corner lots, floor level in high-rises, water facing).
* Inflationary macro-economic interest rate cycles are held static.

## Key Learning Outcomes

* Developing continuous regression models for financial and real estate valuation.
* Comparing ensemble tree regressors (Gradient Boosting & Random Forest) against linear baselines.
* Diagnosing regression residuals and evaluating $R^2$, RMSE, and MAE.
* Creating interactive property valuation estimators with real-time UI controls.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
