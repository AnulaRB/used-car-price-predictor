import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

MODEL_PATH = "model.joblib"
REFERENCE_YEAR = 2026  # same year the notebook used to compute `age`

# Same tiering rule as get_brand_tier() in the notebook (anything else = Economy).
LUXURY = {"BMW", "Mercedes-Benz", "Audi", "Lexus", "Land Rover", "Volvo", "Cadillac", "Lincoln"}
EXOTIC = {"Porsche", "Ferrari", "Lamborghini", "Maserati", "Aston Martin"}


def get_brand_tier(brand: str) -> str:
    if brand in LUXURY:
        return "Luxury"
    if brand in EXOTIC:
        return "Exotic"
    return "Economy"


BRANDS = sorted([
    "Acura", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Buick", "Cadillac",
    "Chevrolet", "Chrysler", "Dodge", "Ferrari", "Fiat", "Ford", "Genesis", "GMC", "Honda",
    "Hummer", "Hyundai", "INFINITI", "Jaguar", "Jeep", "Kia", "Lamborghini", "Land Rover",
    "Lexus", "Lincoln", "Lotus", "Maserati", "Mazda", "McLaren", "Mercedes-Benz", "MINI",
    "Mitsubishi", "Nissan", "Pontiac", "Porsche", "RAM", "Rivian", "Rolls-Royce", "Saab",
    "Saturn", "Scion", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
])
# Categories the notebook's OneHotEncoder actually saw for these columns.
FUEL_TYPES = ["Gasoline", "Diesel", "Hybrid", "E85 Flex Fuel", "Plug-In Hybrid", "Unknown"]
TRANSMISSIONS = ["Automatic", "Manual"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_features(inputs: dict, model) -> pd.DataFrame:
    """Build exactly the 9 columns the notebook's pipeline was trained on."""
    row = {
        "age": REFERENCE_YEAR - inputs["model_year"],
        "milage": float(inputs["mileage"]),  # the dataset spells it "milage"
        "hp": float(inputs["hp"]),
        "engine_l": float(inputs["engine_l"]),
        "cylinders": float(inputs["cylinders"]),
        "fuel_type": inputs["fuel_type"],
        "transmission": inputs["transmission"],
        "brand_tier": get_brand_tier(inputs["brand"]),
        "accident": int(inputs["accident"]),  # 1 = accident/damage reported, 0 = none
    }
    df = pd.DataFrame([row])

    # Safety net: align to the columns the saved pipeline was fitted on.
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"The model expects columns the app doesn't provide: {missing}. "
                "Add them in build_features() in app.py."
            )
        df = df[list(expected)]
    return df


# ------------------------------- UI ----------------------------------------
st.title("🚗 Used Car Price Predictor")
st.caption("Estimate the resale price of a pre-owned vehicle from its specs.")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Export your trained pipeline from the notebook "
        "with `joblib.dump(gb_pipeline, 'model.joblib')` and place it next to app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    usd_to_inr = st.number_input("USD → INR rate", min_value=50.0, max_value=150.0, value=85.0, step=0.5)

st.subheader("Vehicle details")
c1, c2 = st.columns(2)
with c1:
    brand = st.selectbox("Brand", BRANDS, index=None, placeholder="Select a brand")
    model_year = st.number_input(
        "Model year", min_value=1990, max_value=2025, value=None, step=1, placeholder="e.g. 2018"
    )
    mileage = st.number_input(
        "Mileage (miles)", min_value=0, max_value=400_000, value=None, step=1_000, placeholder="e.g. 50000"
    )
    fuel_type = st.selectbox("Fuel type", FUEL_TYPES, index=None, placeholder="Select fuel type")
    transmission = st.selectbox("Transmission", TRANSMISSIONS, index=None, placeholder="Select transmission")
with c2:
    hp = st.number_input(
        "Horsepower (HP)", min_value=50, max_value=1_000, value=None, step=10, placeholder="e.g. 200"
    )
    engine_l = st.number_input(
        "Engine size (L)", min_value=0.6, max_value=8.5, value=None, step=0.1, format="%.1f",
        placeholder="e.g. 2.0",
    )
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=None, placeholder="Select cylinders")
    accident = st.radio("Accident history", ["None reported", "Accident or damage reported"], index=None)

if st.button("Predict price", type="primary"):
    required = {
        "Brand": brand, "Model year": model_year, "Mileage": mileage, "Fuel type": fuel_type,
        "Transmission": transmission, "Horsepower": hp, "Engine size": engine_l,
        "Cylinders": cylinders, "Accident history": accident,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        st.warning("Please fill in: " + ", ".join(missing))
        st.stop()

    inputs = dict(
        brand=brand, model_year=model_year, mileage=mileage, fuel_type=fuel_type,
        transmission=transmission, hp=hp, engine_l=engine_l, cylinders=cylinders,
        accident=accident != "None reported",
    )
    try:
        X = build_features(inputs, model)
        pred_log = float(model.predict(X)[0])
        price_usd = max(float(np.expm1(pred_log)), 0.0)  # model was trained on log1p(price)

        st.success("Prediction complete")
        m1, m2 = st.columns(2)
        m1.metric("Estimated price (USD)", f"${price_usd:,.0f}")
        m2.metric("Approx. price (INR)", f"₹{price_usd * usd_to_inr:,.0f}")
        st.caption(
            "Training data is US-based, so this is a US-market estimate. "
            "Treat it as a rough guide, especially for rare or exotic models."
        )
        with st.expander("Features sent to the model"):
            st.dataframe(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")