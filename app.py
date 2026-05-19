import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://raw.githubusercontent.com/rdalvarez/CoderHouse-DataScienceII/refs/heads/main/EV_Usage_Dataset_v2.csv"


st.set_page_config(
    page_title="EV Range Anxiety Risk",
    page_icon="🔋",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["range_anxiety_risk"]).copy()
    df["trip_date"] = pd.to_datetime(df["trip_date"], errors="coerce")
    df["range_anxiety_risk"] = df["range_anxiety_risk"].astype(int)

    df["trip_year"] = df["trip_date"].dt.year
    df["trip_month"] = df["trip_date"].dt.month
    df["trip_dayofweek"] = df["trip_date"].dt.dayofweek
    df["energy_per_km"] = df["energy_consumed_kwh"] / df["distance_km"]
    df["energy_per_km"] = df["energy_per_km"].replace([np.inf, -np.inf], np.nan)

    return df


def split_features(df):
    target = "range_anxiety_risk"
    columns_to_drop = ["user_id", "trip_date", "vehicle_type_label", "state", "year", "month", target]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_features, categorical_features


@st.cache_resource
def train_model(df):
    X, y, numeric_features, categorical_features = split_features(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    return pipeline, metrics, X


def plot_target_distribution(df):
    counts = df["range_anxiety_risk"].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Distribucion de riesgo de ansiedad por autonomia")
    ax.set_xlabel("Riesgo")
    ax.set_ylabel("Cantidad de viajes")
    ax.set_xticklabels(["Bajo", "Alto"], rotation=0)
    return fig


def plot_risk_by_vehicle(df):
    risk_by_vehicle = df.groupby("vehicle_type")["range_anxiety_risk"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    risk_by_vehicle.plot(kind="bar", ax=ax)
    ax.set_title("Proporcion de alto riesgo por tipo de vehiculo")
    ax.set_xlabel("Tipo de vehiculo")
    ax.set_ylabel("Proporcion de alto riesgo")
    ax.tick_params(axis="x", rotation=0)
    return fig


df = load_data()
model, metrics, X_reference = train_model(df)

st.title("Prediccion de ansiedad por autonomia en vehiculos electricos")
st.write(
    "Esta app usa el mismo dataset del proyecto final para estimar si un viaje puede tener alto riesgo de ansiedad por autonomia."
)

st.subheader("Resumen del dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{df.shape[0]:,}".replace(",", "."))
col2.metric("Columnas", df.shape[1])
col3.metric("Alto riesgo", f"{df['range_anxiety_risk'].mean() * 100:.1f}%")

st.subheader("Metricas del modelo")
metric_cols = st.columns(len(metrics))
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

st.subheader("Analisis visual")
left, right = st.columns(2)
with left:
    st.pyplot(plot_target_distribution(df))
with right:
    st.pyplot(plot_risk_by_vehicle(df))

st.subheader("Prediccion manual")
st.write("Completa los datos principales del viaje para estimar el riesgo.")

with st.form("prediction_form"):
    col_a, col_b, col_c = st.columns(3)

    city = col_a.selectbox("Ciudad", sorted(df["city"].dropna().unique()))
    vehicle_type = col_b.selectbox("Tipo de vehiculo", sorted(df["vehicle_type"].dropna().unique()))
    charging_type = col_c.selectbox("Tipo de carga", sorted(df["charging_type"].dropna().unique()))

    col_d, col_e, col_f = st.columns(3)
    weather_condition = col_d.selectbox("Clima", sorted(df["weather_condition"].dropna().unique()))
    user_income_level = col_e.selectbox("Nivel de ingreso", sorted(df["user_income_level"].dropna().unique()))
    trip_month = col_f.slider("Mes del viaje", 1, 12, 6)

    col_g, col_h, col_i = st.columns(3)
    battery_capacity_kwh = col_g.number_input("Capacidad bateria kWh", 1.0, 60.0, 4.0)
    battery_health_pct = col_h.number_input("Salud bateria %", 50.0, 120.0, 90.0)
    distance_km = col_i.number_input("Distancia km", 1.0, 200.0, 25.0)

    col_j, col_k, col_l = st.columns(3)
    daily_trip_count = col_j.number_input("Viajes diarios", 1.0, 10.0, 2.0)
    charging_frequency_per_week = col_k.number_input("Cargas por semana", 1.0, 10.0, 3.0)
    charging_duration_min = col_l.number_input("Duracion carga min", 0.0, 500.0, 180.0)

    col_m, col_n, col_o = st.columns(3)
    charging_station_distance_km = col_m.number_input("Distancia a estacion km", 0.0, 20.0, 2.0)
    energy_consumed_kwh = col_n.number_input("Energia consumida kWh", 0.1, 25.0, 4.0)
    electricity_cost_per_kwh = col_o.number_input("Costo electricidad", 1.0, 15.0, 8.0)

    col_p, col_q = st.columns(2)
    traffic_density = col_p.slider("Densidad de trafico", 0.0, 1.0, 0.3)
    range_km_estimated = col_q.number_input("Autonomia estimada km", 1.0, 350.0, 40.0)

    submitted = st.form_submit_button("Predecir riesgo")

if submitted:
    energy_per_km = energy_consumed_kwh / distance_km if distance_km else np.nan

    input_data = pd.DataFrame(
        [
            {
                "battery_capacity_kwh": battery_capacity_kwh,
                "battery_health_pct": battery_health_pct,
                "distance_km": distance_km,
                "daily_trip_count": daily_trip_count,
                "charging_frequency_per_week": charging_frequency_per_week,
                "charging_duration_min": charging_duration_min,
                "charging_station_distance_km": charging_station_distance_km,
                "energy_consumed_kwh": energy_consumed_kwh,
                "electricity_cost_per_kwh": electricity_cost_per_kwh,
                "traffic_density": traffic_density,
                "range_km_estimated": range_km_estimated,
                "trip_year": 2026,
                "trip_month": trip_month,
                "trip_dayofweek": 2,
                "energy_per_km": energy_per_km,
                "city": city,
                "vehicle_type": vehicle_type,
                "charging_type": charging_type,
                "weather_condition": weather_condition,
                "user_income_level": user_income_level,
            }
        ]
    )

    probability = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(f"Alto riesgo de ansiedad por autonomia. Probabilidad estimada: {probability:.2%}")
    else:
        st.success(f"Bajo riesgo de ansiedad por autonomia. Probabilidad estimada: {probability:.2%}")

st.caption("Nota: el dataset es sintetico, por lo que los resultados pueden ser mejores que en un caso real.")
