from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import the custom class module before loading the Joblib artifact.
# The saved model references fraud_features.ResampledClassifier.
import fraud_features  # noqa: F401

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Page configuration and visual theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartPay | Mobile Wallet Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: #F5F7FB; }
      [data-testid="stHeader"] { background: rgba(245,247,251,0.92); }
      [data-testid="stSidebar"] { background: #101B32; }
      [data-testid="stSidebar"] * { color: #EAF0FA !important; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 7px 5px; }
      h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #14213D; }
      h1 { letter-spacing: -1.2px; }
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
      .hero {
        background: linear-gradient(120deg, #101B32 0%, #19365C 65%, #235B78 100%);
        padding: 28px 30px; border-radius: 18px; color: white; margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(16,27,50,0.12);
      }
      .hero h1 { color: white !important; margin: 0 0 6px 0; font-size: 2.0rem; }
      .hero p { color: #D4E1F2; margin: 0; font-size: 1rem; }
      .eyebrow { color: #83D6D0; text-transform: uppercase; letter-spacing: 1.8px; font-size: .72rem; font-weight: 700; }
      .section-kicker { color: #62718A; text-transform: uppercase; letter-spacing: 1.4px; font-size: .72rem; font-weight: 700; margin-bottom: 5px; }
      .info-card {
        background: white; padding: 18px 20px; border-radius: 14px;
        border: 1px solid #E5EAF2; box-shadow: 0 4px 14px rgba(24,43,77,.04);
      }
      .info-card h3 { margin-top: 0; font-size: 1rem; }
      .muted { color: #68778F; font-size: .88rem; }
      .risk-high { background: #FFF0EF; border: 1px solid #FFD4D0; color: #A52A25; padding: 13px 16px; border-radius: 12px; font-weight: 700; }
      .risk-low { background: #EAF8F1; border: 1px solid #C9EBDD; color: #176846; padding: 13px 16px; border-radius: 12px; font-weight: 700; }
      .risk-medium { background: #FFF7E8; border: 1px solid #F5E1B8; color: #8C5A08; padding: 13px 16px; border-radius: 12px; font-weight: 700; }
      div[data-testid="stMetric"] { background: white; padding: 14px 16px; border: 1px solid #E5EAF2; border-radius: 13px; }
      div[data-testid="stMetricLabel"] { color: #68778F; }
      div[data-testid="stMetricValue"] { color: #14213D; }
      div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; border: 0; font-weight: 700; min-height: 2.7rem;
      }
      div.stButton > button[kind="primary"] { background: #176B72; }
      div[data-testid="stFileUploader"] { background: white; border-radius: 12px; }
      hr { border-color: #E5EAF2; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load model artifact
# -----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "fraud_pipeline.joblib"


@st.cache_resource(show_spinner="Loading the trained fraud detection model...")
def load_artifact(model_path: str):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "The model file 'fraud_pipeline.joblib' was not found. "
        "Place it in the same folder as App.py."
    )
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
    model = artifact["pipeline"]
    INPUT_COLUMNS = list(artifact["input_columns"])
    SUPPORTED_TYPES = list(artifact["in_scope_types"])
    DEFAULT_THRESHOLD = float(artifact["threshold"])
    MODEL_NAME = str(artifact["model_name"])
    COST_ASSUMPTIONS = artifact.get(
        "cost_assumptions", {"fn_loss_fraction": 1.0, "fp_cost": 500.0}
    )
    FRAUD_INDEX = list(model.classes_).index(1)
except Exception as exc:
    st.error("The saved model could not be loaded.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def hero(title: str, subtitle: str, eyebrow: str = "SMARTPAY · MOBILE WALLET FRAUD DETECTION"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the exact six model features in training order."""
    missing = [col for col in INPUT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    X = frame[INPUT_COLUMNS].copy()
    if X["type"].isna().any():
        raise ValueError("The transaction type column contains empty values.")
    X["type"] = X["type"].astype(str).str.strip().str.upper()

    invalid_types = sorted(set(X["type"].dropna().unique()) - set(SUPPORTED_TYPES))
    if invalid_types:
        raise ValueError(
            "Unsupported transaction type(s): " + ", ".join(invalid_types)
            + ". This model supports CASH_OUT and TRANSFER only."
        )

    numeric_columns = [col for col in INPUT_COLUMNS if col != "type"]
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        # The fitted pipeline includes imputers; missing numeric values can be handled there.
        finite_values = X[col].dropna()
        if not np.isfinite(finite_values.to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains infinite or invalid values.")
        if (finite_values < 0).any():
            raise ValueError(f"Column '{col}' contains negative values; check the source data.")
    return X[INPUT_COLUMNS]


def fraud_probabilities(X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, FRAUD_INDEX]


def risk_label(probability: float, threshold: float) -> tuple[str, str]:
    if probability >= threshold:
        return "High · Flag for review", "risk-high"
    if probability >= min(0.30, threshold):
        return "Medium · Review recommended", "risk-medium"
    return "Low · Not flagged", "risk-low"


def evaluate_labeled_data(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float):
    if "isFraud" not in frame.columns:
        return None
    y = pd.to_numeric(frame["isFraud"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("If included, isFraud must contain only 0 and 1 labels.")
    y = y.astype(int).to_numpy()
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR-AUC": average_precision_score(y, probabilities) if len(np.unique(y)) > 1 else np.nan,
        "ROC-AUC": roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else np.nan,
    }
    return y, pred, metrics


def add_predictions(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float):
    results = frame.copy()
    results["fraud_probability"] = probabilities
    results["fraud_probability_percent"] = (probabilities * 100).round(2)
    results["prediction"] = np.where(probabilities >= threshold, "Potential fraud", "Not flagged")
    results["risk_level"] = [risk_label(float(p), threshold)[0] for p in probabilities]
    return results


def cost_summary(frame: pd.DataFrame, evaluation):
    y, pred, metrics = evaluation
    if "amount" not in frame.columns:
        raise ValueError("The amount column is required for the missed-fraud cost calculation.")
    amounts = pd.to_numeric(frame["amount"], errors="coerce").fillna(0).to_numpy(dtype=float)
    fn_mask = (y == 1) & (pred == 0)
    fp_mask = (y == 0) & (pred == 1)
    fn_cost = float(COST_ASSUMPTIONS.get("fn_loss_fraction", 1.0)) * float(amounts[fn_mask].sum())
    fp_cost = float(COST_ASSUMPTIONS.get("fp_cost", 500.0)) * int(fp_mask.sum())
    return fn_cost, fp_cost, fn_cost + fp_cost


# -----------------------------------------------------------------------------
# Sidebar navigation and threshold
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='padding:8px 4px 18px 4px;'>"
    "<div style='font-size:1.4rem;font-weight:800;color:white;'>💳 SmartPay</div>"
    "<div style='font-size:.78rem;color:#AFC2DA;'>MOBILE WALLET FRAUD DETECTION</div></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Single Transaction", "Batch Analysis", "Model Performance"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("### Decision settings")
threshold = st.sidebar.slider(
    "Fraud threshold",
    min_value=0.05,
    max_value=0.99,
    value=float(np.clip(DEFAULT_THRESHOLD, 0.05, 0.99)),
    step=0.01,
    help="Transactions at or above this probability are flagged for review.",
)
st.sidebar.caption(f"Saved validation-selected threshold: {DEFAULT_THRESHOLD:.2f}")
st.sidebar.caption(f"Current threshold: **{threshold:.2f}**")
st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.write(MODEL_NAME)
st.sidebar.caption("Supports CASH_OUT and TRANSFER transactions.")
st.sidebar.markdown("[GitHub repository](https://github.com/Shahd-Salah/Mobile-Wallet-Fraud-Detection)")


# -----------------------------------------------------------------------------
# Page: SmartPay overview
# -----------------------------------------------------------------------------
if page == "Overview":
    hero(
        "SmartPay — Mobile Wallet Fraud Detection",
        "A machine-learning decision-support tool for identifying potentially suspicious wallet transfers and cash-outs.",
    )
    st.markdown('<div class="section-kicker">Model benchmark · chronological holdout test</div>', unsafe_allow_html=True)
    st.subheader("Verified test performance")
    st.caption(
        "Results from the notebook's untouched test period (steps 632–743), "
        "calculated at the deployed 0.70 threshold. PaySim is simulated data."
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Test transactions", "37,979")
    k2.metric("Precision", "100.00%")
    k3.metric("Recall", "99.60%")
    k4.metric("F1 score", "99.80%")
    k5.metric("PR-AUC", "1.000")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("#### Test confusion matrix")
        confusion_df = pd.DataFrame(
            [[36727, 0], [5, 1247]],
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"],
        )
        st.dataframe(confusion_df, use_container_width=True)
        st.caption("At threshold 0.70: 1,247 fraud cases detected, 5 missed, and 0 legitimate transactions falsely flagged.")
    with right:
        st.markdown("#### Decision policy")
        st.markdown(
            f"<div class='info-card'><div class='section-kicker'>Current threshold</div>"
            f"<div style='font-size:2.3rem;font-weight:800;color:#176B72;'>{threshold:.0%}</div>"
            "<div class='muted'>A transaction at or above this score is flagged for review. "
            "A flag is not confirmation of fraud.</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            "<div class='info-card'><h3>Model coverage</h3>"
            "<div style='font-size:1.15rem;font-weight:700;color:#14213D;'>CASH_OUT · TRANSFER</div>"
            "<div class='muted'>The model is intentionally scoped to the two transaction types containing fraud labels in the supplied PaySim dataset.</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="section-kicker">Dataset insights</div>', unsafe_allow_html=True)
    st.subheader("Where fraud appeared in the supplied dataset")
    fraud_type = pd.DataFrame(
        {"Fraud rate (%)": [0.183955, 0.768799]},
        index=["CASH_OUT", "TRANSFER"],
    )
    st.bar_chart(fraud_type, y="Fraud rate (%)", color="#176B72")
    st.caption("Full-dataset aggregate fraud rates: CASH_OUT 0.1840%; TRANSFER 0.7688%. Other transaction types had no fraud labels in this simulated dataset.")

    st.subheader("Most influential validation signals")
    importance = pd.DataFrame(
        {"Mean PR-AUC drop": [0.9271, 0.4710, 0.0553, 0.0108, 0.0002, 0.0000]},
        index=["oldbalanceOrg", "amount", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type"],
    )
    st.bar_chart(importance, y="Mean PR-AUC drop", color="#367B8E")
    st.caption("Permutation importance on validation data. These are model associations, not causal explanations for an individual transaction.")

    st.info(
        "This benchmark is based on simulated PaySim records. Its unusually strong results should not be assumed to transfer directly to real wallet-provider traffic."
    )

    with st.expander("Project information & responsible use"):
        info_left, info_right = st.columns(2, gap="large")
        with info_left:
            st.markdown("**Dataset**")
            st.write("PaySim Mobile Money Simulation · 6,362,620 transactions · 8,213 fraud labels · 0.1291% overall fraud prevalence.")
            st.markdown("**Modeling workflow**")
            st.write("Chronological split → training-only sampling and preprocessing → imbalance comparison → model comparison → time-aware cross-validation → validation threshold selection → untouched test evaluation.")
            st.markdown("**Deployed model**")
            st.write(f"{MODEL_NAME}, six input features, and a validation-selected threshold of {DEFAULT_THRESHOLD:.2f}.")
        with info_right:
            st.markdown("**Supported transaction types**")
            st.write("CASH_OUT and TRANSFER only.")
            st.markdown("**Cost assumptions**")
            st.write("Missed-fraud loss: 100% of transaction amount. False-alarm cost: 500 illustrative monetary units per flagged legitimate transaction. These are not verified EGP operational costs.")
            st.markdown("**Responsible use**")
            st.write("A prediction is a risk signal, not proof of fraud. Real payment decisions require verification, human review, monitoring, and policy controls.")
        st.markdown("**Project links**")
        st.markdown("- [Live application](https://mobile-wallet-fraud-detection.streamlit.app/)\n- [GitHub repository](https://github.com/Shahd-Salah/Mobile-Wallet-Fraud-Detection)\n- [PaySim dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)")


# -----------------------------------------------------------------------------
# Page: Single transaction
# -----------------------------------------------------------------------------
elif page == "Single Transaction":
    hero(
        "Single Transaction Screening",
        "Enter a transaction's available balance and payment details to receive a fraud-risk score.",
    )
    st.markdown('<div class="section-kicker">Transaction input</div>', unsafe_allow_html=True)
    st.subheader("Transaction details")
    with st.form("single_transaction_form", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            tx_type = st.selectbox("Transaction type", SUPPORTED_TYPES, help="This model supports CASH_OUT and TRANSFER only.")
            amount = st.number_input("Transaction amount", min_value=0.0, value=1000.0, step=100.0)
            old_org = st.number_input("Sender balance before transaction", min_value=0.0, value=5000.0, step=100.0)
        with col2:
            new_org = st.number_input("Sender balance after transaction", min_value=0.0, value=4000.0, step=100.0)
            old_dest = st.number_input("Receiver balance before transaction", min_value=0.0, value=1000.0, step=100.0)
            new_dest = st.number_input("Receiver balance after transaction", min_value=0.0, value=2000.0, step=100.0)
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)

    if submitted:
        transaction = pd.DataFrame([{
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
        }])
        try:
            X = make_features(transaction)
            probability = float(fraud_probabilities(X)[0])
            label, css_class = risk_label(probability, threshold)
            st.markdown("### Screening result")
            a, b = st.columns([1, 1.4])
            with a:
                st.metric("Predicted fraud probability", f"{probability:.2%}")
                st.progress(float(np.clip(probability, 0, 1)))
            with b:
                st.markdown(f"<div class='{css_class}'>{label}</div>", unsafe_allow_html=True)
                st.caption(f"Current decision threshold: {threshold:.0%}. This is a model prediction, not a confirmed fraud finding.")
            with st.expander("Review submitted inputs"):
                st.dataframe(transaction, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


# -----------------------------------------------------------------------------
# Page: Batch analysis
# -----------------------------------------------------------------------------
elif page == "Batch Analysis":
    hero(
        "Batch Transaction Analysis",
        "Upload a transaction CSV, score every supported record, inspect risk groups, and download the results.",
    )
    st.markdown('<div class="section-kicker">Batch screening</div>', unsafe_allow_html=True)
    st.subheader("Upload transaction data")
    st.caption("Required columns: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest. Optional evaluation label: isFraud.")

    template = pd.DataFrame(columns=INPUT_COLUMNS)
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="wallet_transaction_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")

    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
            if raw.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                X = make_features(raw)
                probabilities = fraud_probabilities(X)
                results = add_predictions(raw, probabilities, threshold)
                st.session_state["latest_batch_raw"] = raw.copy()
                st.session_state["latest_batch_results"] = results.copy()
                st.session_state["latest_batch_probabilities"] = probabilities.copy()

                flagged = int((probabilities >= threshold).sum())
                average_risk = float(np.mean(probabilities))
                a, b, c, d = st.columns(4)
                a.metric("Transactions analyzed", f"{len(results):,}")
                b.metric("Flagged for review", f"{flagged:,}")
                c.metric("Not flagged", f"{len(results) - flagged:,}")
                d.metric("Average fraud score", f"{average_risk:.2%}")

                st.markdown("#### Risk distribution")
                risk_counts = results["risk_level"].value_counts().reindex(
                    ["Low · Not flagged", "Medium · Review recommended", "High · Flag for review"], fill_value=0
                )
                st.bar_chart(risk_counts, color="#176B72")

                st.markdown("#### Scored transactions")
                display_columns = [c for c in results.columns if c != "fraud_probability"]
                st.dataframe(results[display_columns], use_container_width=True, height=360)
                st.download_button(
                    "Download scored transactions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="wallet_fraud_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

                if "isFraud" in raw.columns:
                    st.divider()
                    st.markdown("#### Uploaded-file evaluation")
                    st.caption("These metrics describe only this uploaded file. Use the notebook's untouched chronological test set for official project performance.")
                    evaluation = evaluate_labeled_data(raw, probabilities, threshold)
                    y, pred, metrics = evaluation
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Precision", f"{metrics['Precision']:.2%}")
                    p2.metric("Recall", f"{metrics['Recall']:.2%}")
                    p3.metric("F1", f"{metrics['F1']:.2%}")
                    p4.metric("PR-AUC", f"{metrics['PR-AUC']:.4f}" if np.isfinite(metrics["PR-AUC"]) else "N/A")
                    st.dataframe(
                        pd.DataFrame(
                            [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
                            index=["Actual legitimate", "Actual fraud"],
                            columns=["Predicted legitimate", "Predicted fraud"],
                        ),
                        use_container_width=True,
                    )
                    fn_cost, fp_cost, total_cost = cost_summary(raw, evaluation)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Missed-fraud cost", f"{fn_cost:,.2f}")
                    c2.metric("False-alarm cost", f"{fp_cost:,.2f}")
                    c3.metric("Estimated total cost", f"{total_cost:,.2f}")
                    st.caption("Illustrative PaySim monetary units: missed fraud = transaction amount × loss fraction; false alarm = 500 units. These are not verified EGP operational costs.")
        except Exception as exc:
            st.error("The CSV could not be processed.")
            st.exception(exc)


# -----------------------------------------------------------------------------
# Page: Model performance and threshold explorer
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    hero(
        "Model Performance & Threshold Studio",
        "Review the verified chronological holdout benchmark and explore the threshold trade-off on your own labelled upload.",
    )
    st.markdown('<div class="section-kicker">Official notebook benchmark</div>', unsafe_allow_html=True)
    st.subheader("Untouched test set · threshold 0.70")
    a, b, c, d, e = st.columns(5)
    a.metric("Precision", "100.00%")
    b.metric("Recall", "99.60%")
    c.metric("F1", "99.80%")
    d.metric("ROC-AUC", "1.0000")
    e.metric("PR-AUC", "1.0000")
    benchmark_cm = pd.DataFrame(
        [[36727, 0], [5, 1247]],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    st.dataframe(benchmark_cm, use_container_width=True)
    st.caption("37,979 transactions; 1,252 fraud cases. These are the confirmed notebook results at the selected 0.70 threshold.")

    st.divider()
    st.markdown('<div class="section-kicker">Interactive threshold analysis</div>', unsafe_allow_html=True)
    st.subheader("Explore thresholds using a labelled CSV")
    st.caption("Upload labelled validation data for threshold exploration. Do not use the small 20-row application demo file as a final performance estimate.")
    threshold_file = st.file_uploader("Upload labelled transactions", type=["csv"], key="threshold_csv")
    if threshold_file is not None:
        try:
            raw = pd.read_csv(threshold_file)
            if "isFraud" not in raw.columns:
                st.error("The uploaded file needs an isFraud column for threshold evaluation.")
            else:
                X = make_features(raw)
                y_series = pd.to_numeric(raw["isFraud"], errors="coerce")
                if y_series.isna().any() or not y_series.isin([0, 1]).all():
                    st.error("isFraud must contain only 0 and 1.")
                else:
                    probs = fraud_probabilities(X)
                    y = y_series.astype(int).to_numpy()
                    current_pred = (probs >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, current_pred, labels=[0, 1]).ravel()
                    p = precision_score(y, current_pred, zero_division=0)
                    r = recall_score(y, current_pred, zero_division=0)
                    f = f1_score(y, current_pred, zero_division=0)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Precision at current threshold", f"{p:.2%}")
                    m2.metric("Recall", f"{r:.2%}")
                    m3.metric("False positives", f"{fp:,}")
                    m4.metric("False negatives", f"{fn:,}")

                    amounts = pd.to_numeric(raw["amount"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    fn_loss_fraction = float(COST_ASSUMPTIONS.get("fn_loss_fraction", 1.0))
                    fp_unit_cost = float(COST_ASSUMPTIONS.get("fp_cost", 500.0))
                    threshold_rows = []
                    for t in np.linspace(0.05, 0.99, 60):
                        pred = (probs >= t).astype(int)
                        missed = (y == 1) & (pred == 0)
                        false_alarm = (y == 0) & (pred == 1)
                        total = fn_loss_fraction * amounts[missed].sum() + fp_unit_cost * false_alarm.sum()
                        threshold_rows.append({"Threshold": round(float(t), 3), "Estimated cost": float(total)})
                    curve = pd.DataFrame(threshold_rows).set_index("Threshold")
                    st.markdown("#### Estimated cost by threshold")
                    st.line_chart(curve, y="Estimated cost", color="#176B72")
                    current_fn_cost = fn_loss_fraction * amounts[(y == 1) & (current_pred == 0)].sum()
                    current_fp_cost = fp_unit_cost * int(((y == 0) & (current_pred == 1)).sum())
                    st.metric("Estimated cost at current threshold", f"{current_fn_cost + current_fp_cost:,.2f}")
                    st.caption("This curve is for the uploaded labelled data only. Threshold selection for the final project must remain based on validation data, not the untouched test set.")
        except Exception as exc:
            st.error("Threshold analysis could not be completed.")
            st.exception(exc)



st.divider()
st.markdown(
    "<div style='text-align:center;color:#7B879A;font-size:.78rem;'>"
    "SMARTPAY · Mobile wallet fraud detection · Academic machine-learning demonstration · A flagged transaction requires review and does not confirm fraud."
    "</div>",
    unsafe_allow_html=True,
)
