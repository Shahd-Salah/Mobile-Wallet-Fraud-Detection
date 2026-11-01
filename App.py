
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# SHOW TITLE IMMEDIATELY
st.title("💳 Mobile Wallet Fraud Detection")
st.write("Intelligent fraud detection for mobile-wallet transactions.")

# LOAD MODEL
MODEL_PATH = Path(__file__).parent / "fraud_pipeline.joblib"

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Could not find fraud_pipeline.joblib.")
    st.stop()

try:
    artifact = load_artifact()
    model = artifact["pipeline"]
    columns = artifact["input_columns"]
    transaction_types = artifact["in_scope_types"]
    saved_threshold = float(artifact["threshold"])
    fraud_index = list(model.classes_).index(1)

    st.success("Your fraud detection model loaded successfully!")

except Exception as e:
    st.error("There was a problem loading the model.")
    st.exception(e)
    st.stop()

# SIDEBAR
st.sidebar.title("Navigation")

threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=saved_threshold,
    step=0.01
)

page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Single Transaction", "CSV Upload", "Fraud Dashboard"]
)

# HOME PAGE
if page == "Home":

    st.header("Welcome!")

    st.write(
        "This application uses a trained Logistic Regression model "
        "to identify potentially fraudulent mobile-wallet transactions."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Machine Learning Model", artifact["model_name"])
    col2.metric("Saved Threshold", f"{saved_threshold:.0%}")
    col3.metric("Input Features", len(columns))

    st.subheader("How It Works")

    st.write("1. Enter a transaction or upload a CSV file.")
    st.write("2. The model calculates the probability of fraud.")
    st.write("3. The application compares the probability with the selected threshold.")
    st.write("4. The transaction is flagged for review when its score reaches the threshold.")

# SINGLE TRANSACTION
elif page == "Single Transaction":

    st.header("Single Transaction Detection")

    with st.form("transaction_form"):

        transaction = {}

        transaction["type"] = st.selectbox(
            "Transaction Type",
            transaction_types
        )

        transaction["amount"] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        transaction["oldbalanceOrg"] = st.number_input(
            "Sender Balance Before Transaction",
            min_value=0.0,
            value=5000.0
        )

        transaction["newbalanceOrig"] = st.number_input(
            "Sender Balance After Transaction",
            min_value=0.0,
            value=4000.0
        )

        transaction["oldbalanceDest"] = st.number_input(
            "Receiver Balance Before Transaction",
            min_value=0.0,
            value=1000.0
        )

        transaction["newbalanceDest"] = st.number_input(
            "Receiver Balance After Transaction",
            min_value=0.0,
            value=2000.0
        )

        submitted = st.form_submit_button("Detect Fraud")

    if submitted:

        try:
            X = pd.DataFrame(
                [transaction],
                columns=columns
            )

            probability = model.predict_proba(X)[0][fraud_index]

            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            if probability >= threshold:
                st.error("Potentially Fraudulent Transaction")
                st.write("This transaction should be reviewed.")
            else:
                st.success("Transaction Not Flagged")
                st.write("The fraud score is below the selected threshold.")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# CSV UPLOAD
elif page == "CSV Upload":

    st.header("Multiple Transaction Detection")

    st.write("Upload a CSV file containing your transaction information.")

    template = pd.DataFrame(columns=columns)

    st.download_button(
        "Download CSV Template",
        template.to_csv(index=False),
        "transaction_template.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload Transactions",
        type=["csv"]
    )

    if uploaded_file:

        try:
            data = pd.read_csv(uploaded_file)

            missing = [
                col for col in columns
                if col not in data.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            elif data.empty:
                st.warning("Your file is empty.")

            else:

                X = data[columns].copy()

                for col in columns:
                    if col != "type":
                        X[col] = pd.to_numeric(
                            X[col],
                            errors="coerce"
                        )

                if X.isna().any().any():
                    st.error("Your file contains missing or invalid values.")

                elif not set(X["type"].unique()).issubset(
                    set(transaction_types)
                ):
                    st.error(
                        "Only CASH_OUT and TRANSFER transactions "
                        "are supported by this model."
                    )

                else:

                    probabilities = model.predict_proba(X)[:, fraud_index]

                    results = data.copy()

                    results["Fraud Probability (%)"] = (
                        probabilities * 100
                    ).round(2)

                    results["Prediction"] = np.where(
                        probabilities >= threshold,
                        "Potential Fraud",
                        "Not Flagged"
                    )

                    flagged = int((probabilities >= threshold).sum())

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Transactions", len(results))
                    c2.metric("Potential Fraud", flagged)
                    c3.metric("Not Flagged", len(results) - flagged)

                    st.subheader("Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Results",
                        results.to_csv(index=False),
                        "fraud_predictions.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error("Could not process your CSV.")
            st.exception(e)

# FRAUD DASHBOARD
elif page == "Fraud Dashboard":

    st.header("Fraud Detection Dashboard")

    st.write(
        "Upload labelled validation data to evaluate the model "
        "at the selected threshold."
    )

    validation_file = st.file_uploader(
        "Upload labelled CSV",
        type=["csv"]
    )

    if validation_file:

        try:
            from sklearn.metrics import (
                confusion_matrix,
                precision_score,
                recall_score,
                f1_score,
                average_precision_score
            )

            data = pd.read_csv(validation_file)

            required = columns + ["isFraud"]
            missing = [c for c in required if c not in data.columns]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:

                X = data[columns].copy()
                y = pd.to_numeric(data["isFraud"], errors="coerce")

                for col in columns:
                    if col != "type":
                        X[col] = pd.to_numeric(X[col], errors="coerce")

                if X.isna().any().any() or y.isna().any():
                    st.error("Missing or invalid values found.")

                elif not y.isin([0, 1]).all():
                    st.error("isFraud must contain only 0 and 1.")

                elif not set(X["type"].unique()).issubset(
                    set(transaction_types)
                ):
                    st.error("Unsupported transaction types.")

                else:

                    probabilities = model.predict_proba(X)[:, fraud_index]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    tn, fp, fn, tp = confusion_matrix(
                        y.astype(int),
                        predictions,
                        labels=[0, 1]
                    ).ravel()

                    precision = precision_score(
                        y, predictions, zero_division=0
                    )

                    recall = recall_score(
                        y, predictions, zero_division=0
                    )

                    f1 = f1_score(
                        y, predictions, zero_division=0
                    )

                    pr_auc = (
                        average_precision_score(y, probabilities)
                        if y.nunique() == 2 else None
                    )

                    a, b, c, d = st.columns(4)

                    a.metric("Precision", f"{precision:.3f}")
                    b.metric("Recall", f"{recall:.3f}")
                    c.metric("F1 Score", f"{f1:.3f}")
                    d.metric(
                        "PR-AUC",
                        f"{pr_auc:.3f}" if pr_auc is not None else "N/A"
                    )

                    st.subheader("Confusion Matrix")

                    st.dataframe(
                        pd.DataFrame(
                            [[tn, fp], [fn, tp]],
                            index=["Actual Legitimate", "Actual Fraud"],
                            columns=["Predicted Legitimate", "Predicted Fraud"]
                        )
                    )

                    
                    st.subheader("Business Cost Analysis")

                    st.write(
                     "Costs are calculated using the same assumptions "
                    "as the machine learning notebook."
                     )

                    # Retrieve cost assumptions from the saved model
                    cost_assumptions = artifact["cost_assumptions"]

                    fn_loss_fraction = cost_assumptions["fn_loss_fraction"]
                    fp_cost = cost_assumptions["fp_cost"]

                    # Identify missed fraud and false alarms
                    actual = y.astype(int).to_numpy()

                    missed_fraud_mask = (
                        (actual == 1) & (predictions == 0)
                    )

                    false_alarm_mask = (
                        (actual == 0) & (predictions == 1)
                    )

                    # Calculate the total amount of missed fraud
                    missed_fraud_amount = data.loc[
                        missed_fraud_mask, "amount"
                    ].sum()

                    # Calculate the two cost components
                    missed_fraud_cost = (
                        fn_loss_fraction * missed_fraud_amount
                    )

                    false_alarm_total_cost = (
                        fp_cost * int(false_alarm_mask.sum())
                    )

                    total_cost = (
                        missed_fraud_cost + false_alarm_total_cost
                    )

                    # Display results
                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Missed Fraud Cost",
                        f"{missed_fraud_cost:,.2f}"
                    )

                    c2.metric(
                        "False Alarm Cost",
                        f"{false_alarm_total_cost:,.2f}"
                    )

                    c3.metric(
                        "Estimated Total Cost",
                        f"{total_cost:,.2f}"
                    )

                    st.caption(
                        "Cost assumptions: missed fraud = 100% of transaction "
                        "amount; false alarm = 500 units per transaction. "
                        "These are illustrative PaySim monetary units, not EGP."
                    )

        except Exception as e:
            st.error("Dashboard evaluation failed.")
            st.exception(e)

st.divider()
st.caption(
    "Academic demonstration. A flagged transaction requires review "
    "and does not confirm fraud."
)
