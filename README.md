# 🚗 Used Car Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anularb-used-car-price-predictor-app-i720tp.streamlit.app/)

An end-to-end Machine Learning application that estimates the resale market value of pre-owned vehicles based on mechanical specifications, usage history, and brand tiers. Powered by a Scikit-Learn **Gradient Boosting** pipeline and served through an interactive **Streamlit** web app.

🔗 **Live Demo**: [anularb-used-car-price-predictor-app-i720tp.streamlit.app](https://anularb-used-car-price-predictor-app-i720tp.streamlit.app/)

---

## 📌 Project Overview

Valuing used cars accurately requires accounting for non-linear depreciation patterns, wear and tear, vehicle performance specs, and accident histories. This project provides:
1. **Exploratory Data Analysis (EDA) & Modeling**: Documented in [`proj.ipynb`](proj.ipynb), comparing linear and non-linear regression algorithms.
2. **Production Pipeline**: Preprocessing (`StandardScaler` + `OneHotEncoder`) bundled directly with a trained `GradientBoostingRegressor` inside a reproducible Scikit-Learn `Pipeline`.
3. **Interactive Web App**: A clean, accessible UI built with [Streamlit](app.py) allowing users to input car specs and get immediate USD and INR valuations.

---

## 🏗️ Architecture & Features

### 1. Feature Engineering
The model consumes 9 key vehicle attributes:

| Feature | Type | Description | Preprocessing |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Vehicle age relative to reference year (`2026 - model_year`) | `StandardScaler` |
| `milage` | Numeric | Total miles driven | `StandardScaler` |
| `hp` | Numeric | Engine horsepower | `StandardScaler` |
| `engine_l` | Numeric | Engine displacement in liters (e.g., 2.0L, 3.5L) | `StandardScaler` |
| `cylinders` | Numeric | Number of engine cylinders (3, 4, 6, 8, etc.) | `StandardScaler` |
| `fuel_type` | Categorical | Gasoline, Diesel, Hybrid, Plug-In Hybrid, E85, Unknown | `OneHotEncoder` |
| `transmission` | Categorical | Automatic or Manual | `OneHotEncoder` |
| `brand_tier` | Categorical | Tiered category (`Economy`, `Luxury`, `Exotic`) | `OneHotEncoder` |
| `accident` | Binary | Accident or damage history reported (`0 = None`, `1 = Reported`) | `OneHotEncoder` |

### 2. Machine Learning Pipeline
* **Target Variable**: Trained on $\log(1 + \text{price})$ to stabilize variance, normalize right-skewed pricing distributions, and penalize percentage errors symmetrically.
* **Model Selection**: Multiple models were benchmarked in [`proj.ipynb`](proj.ipynb):
  * **Linear Regression** (Baseline)
  * **Ridge Regression** ($\alpha$ tuned via 5-Fold `GridSearchCV`)
  * **Lasso Regression** ($\alpha$ tuned via 5-Fold `GridSearchCV`)
  * **Gradient Boosting Regressor** (*Selected* – best $R^2$ score and lowest RMSE / MAE)
* **Inference**: Prediction output is transformed back to real dollars via $\text{expm1}(\hat{y})$.

---

## 📁 Repository Structure

```text
used-car-price-predictor/
├── app.py              # Streamlit web application & prediction interface
├── proj.ipynb          # Jupyter Notebook: data cleaning, EDA, benchmark & training
├── model.joblib        # Serialized, ready-to-deploy Scikit-Learn Pipeline
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+ recommended
* `pip` and virtual environment tool (`venv` or `conda`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnulaRB/used-car-price-predictor.git
   cd used-car-price-predictor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The app will start locally at `http://localhost:8501`.

### App Features:
* **Interactive Controls**: Dropdowns for brand, fuel type, transmission, and sliders/number inputs for mileage, engine specs, and year.
* **Dual Currency Conversion**: Real-time conversion between **USD ($)** and **INR (₹)** with an adjustable exchange rate in the sidebar.
* **Model Input Inspection**: Expandable view showing the exact engineered feature vector passed to the pipeline.

---

## 📓 Running the Notebook

To view data distributions, residual plots, feature importances, and retrain the model:

```bash
jupyter notebook proj.ipynb
```

---

## 🗺️ Roadmap & Future Enhancements

- [ ] **Price Confidence Intervals**: Implement quantile regression ($\alpha = 0.1, 0.9$) to provide a fair price range (e.g. Kelley Blue Book style).
- [ ] **SHAP Explainability**: Visual breakdown displaying which factors raised or lowered the car's predicted value.
- [ ] **Model Presets**: Automatic specification pre-fill when selecting common car models (e.g. Honda Civic, BMW 3-Series).
- [ ] **Deal Rating Analyzer**: Compare actual seller listing prices against fair market value to flag *Great Deal*, *Fair Deal*, or *Overpriced*.
- [ ] **Advanced Estimators**: Benchmark CatBoost / LightGBM for native high-cardinality categorical support.

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify for personal or commercial projects.
