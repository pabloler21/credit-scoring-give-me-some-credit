import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Credit Scoring - Give Me Some Credit", layout="centered")

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("models/scaler.pkl")
        log_model = joblib.load("models/log_model.pkl")
        rf_model = joblib.load("models/rf_model.pkl")
        return scaler, log_model, rf_model
    except FileNotFoundError as e:
        st.error(
            "No encuentro los archivos en la carpeta `models/`.\n\n"
            "Asegurate de tener:\n"
            "- models/scaler.pkl\n"
            "- models/log_model.pkl\n"
            "- models/rf_model.pkl\n"
        )
        st.stop()

scaler, log_model, rf_model = load_artifacts()

st.title("Credit Scoring App")
st.write("Estimación de probabilidad de morosidad (SeriousDlqin2yrs) con modelos entrenados en *Give Me Some Credit*.")
st.caption("⚠️ Demo educativa para portfolio. No usar como decisión real de crédito.")

# Sidebar: controles globales
st.sidebar.header("Configuración")
model_choice = st.sidebar.selectbox("Modelo", ["Regresión Logística", "Random Forest"])
threshold = st.sidebar.slider("Threshold (umbral de riesgo)", 0.0, 1.0, 0.40, 0.01)

model = log_model if model_choice == "Regresión Logística" else rf_model

st.subheader("Ingresar datos del cliente")

with st.form("client_form"):
    RevolvingUtilizationOfUnsecuredLines = st.number_input(
        "Revolving Utilization (puede ser >1)",
        min_value=0.0, max_value=10.0, value=0.5, step=0.01
    )
    age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)

    NumberOfTime30_59DaysPastDueNotWorse = st.number_input(
        "Veces 30-59 días de mora", min_value=0, max_value=20, value=0, step=1
    )
    NumberOfTime60_89DaysPastDueNotWorse = st.number_input(
        "Veces 60-89 días de mora", min_value=0, max_value=20, value=0, step=1
    )
    NumberOfTimes90DaysLate = st.number_input(
        "Veces 90+ días de mora", min_value=0, max_value=20, value=0, step=1
    )

    DebtRatio = st.number_input(
        "Debt Ratio", min_value=0.0, max_value=10.0, value=0.3, step=0.01
    )
    MonthlyIncome = st.number_input(
        "Ingreso mensual", min_value=0.0, max_value=200000.0, value=4000.0, step=100.0
    )

    NumberOfOpenCreditLinesAndLoans = st.number_input(
        "Líneas de crédito / préstamos abiertos", min_value=0, max_value=50, value=5, step=1
    )
    NumberRealEstateLoansOrLines = st.number_input(
        "Préstamos / líneas inmobiliarias", min_value=0, max_value=20, value=1, step=1
    )
    NumberOfDependents = st.number_input(
        "Dependientes", min_value=0, max_value=20, value=0, step=1
    )

    submitted = st.form_submit_button("Calcular score")

if submitted:
    client_data = {
        'RevolvingUtilizationOfUnsecuredLines': RevolvingUtilizationOfUnsecuredLines,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30_59DaysPastDueNotWorse,
        'DebtRatio': DebtRatio,
        'MonthlyIncome': MonthlyIncome,
        'NumberOfOpenCreditLinesAndLoans': NumberOfOpenCreditLinesAndLoans,
        'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
        'NumberRealEstateLoansOrLines': NumberRealEstateLoansOrLines,
        'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60_89DaysPastDueNotWorse,
        'NumberOfDependents': NumberOfDependents
    }

    df_client = pd.DataFrame([client_data])[FEATURES]  # asegura orden de columnas
    X_scaled = scaler.transform(df_client)             # útil para LR y explicabilidad

    # Consistencia: LR usa escalado, RF usa valores crudos
    X_for_model = X_scaled if model is log_model else df_client.values

    proba = model.predict_proba(X_for_model)[0, 1]
    decision = "RIESGOSO" if proba >= threshold else "NO RIESGOSO"

    col1, col2 = st.columns(2)
    col1.metric("Probabilidad de morosidad", f"{proba:.3f}")
    col2.metric("Probabilidad (%)", f"{proba*100:.1f}%")

    if decision == "RIESGOSO":
        st.error(f"Decisión: {decision} (proba ≥ {threshold:.2f})")
    else:
        st.success(f"Decisión: {decision} (proba < {threshold:.2f})")

    # Explicabilidad: solo para Regresión Logística
    if model is log_model:
        coefs = log_model.coef_.ravel()
        contrib = X_scaled[0] * coefs

        explain_df = pd.DataFrame({
            "feature": FEATURES,
            "input_value": df_client.iloc[0].values,
            "contribution": contrib
        })
        explain_df["abs_contribution"] = explain_df["contribution"].abs()

        explain_df["impact"] = explain_df["contribution"].apply(
        lambda x: "↑ Aumenta riesgo" if x > 0 else "↓ Reduce riesgo"
    )
        explain_df["abs_contribution"] = explain_df["contribution"].abs()
        top = explain_df.sort_values("abs_contribution", ascending=False).head(5)

        st.subheader("Top factores (Regresión Logística)")
        st.caption("Impacto aproximado de cada variable en la predicción.")

        st.dataframe(
        top[["feature", "input_value", "impact", "contribution"]],
        use_container_width=True
)
    else:
        st.info("Top factores: disponible solo para Regresión Logística (coeficientes interpretables).")

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(df_client, use_container_width=True)
    top_features_text = ", ".join(top["feature"].values)

    st.write(
        f"Las variables que más influyeron en la decisión fueron: {top_features_text}."
    )