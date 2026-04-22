import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------- CSS --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 55%, #6366f1 100%);
        padding: 42px 40px;
        border-radius: 28px;
        color: white;
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.20);
        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        line-height: 1.8;
        opacity: 0.96;
        max-width: 900px;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 8px 0 12px 0;
    }

    .card {
        background: rgba(255,255,255,0.96);
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }

    .muted {
        color: #667085;
        line-height: 1.8;
        font-size: 0.96rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.98);
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        text-align: left;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .result-good {
        border-left: 6px solid #16a34a;
    }

    .result-risk {
        border-left: 6px solid #dc2626;
    }

    .result-title-good {
        color: #15803d;
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .result-title-risk {
        color: #dc2626;
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .summary-grid {
        line-height: 2;
        color: #475569;
        font-size: 0.97rem;
    }

    .summary-grid b {
        color: #0f172a;
    }

    .note {
        background: #eef4ff;
        border: 1px solid #cfe0ff;
        color: #1e3a8a;
        border-radius: 16px;
        padding: 14px 16px;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.92rem;
        padding: 8px 0 4px 0;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        height: 3rem;
        font-weight: 700;
        font-size: 1rem;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
    }

    .stButton > button:hover {
        filter: brightness(1.03);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- DATA --------------------
DATA_FILE = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = "models/artifacts/logistic_churn_model.joblib"


@st.cache_data
def load_data(data_file: str = DATA_FILE):
    if not os.path.exists(data_file):
        st.error(
            f"Dataset file not found: {data_file}\n\n"
            f"Please place the CSV file in the same folder as Web.py."
        )
        st.stop()

    df = pd.read_csv(data_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@st.cache_resource
def train_model(df):
    model_df = df.copy()

    # Features chosen to match your form
    features = ["gender", "tenure", "Contract", "TotalCharges"]
    target = "Churn"

    model_df = model_df[features + [target]].dropna()
    model_df[target] = model_df[target].map({"Yes": 1, "No": 0})

    X = model_df[features]
    y = model_df[target]

    categorical_features = ["gender", "Contract"]
    numeric_features = ["tenure", "TotalCharges"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return model, accuracy, auc


@st.cache_resource
def load_or_train_model(df):
    if not os.path.exists(MODEL_FILE):
        st.error(
            f"Saved model not found: {MODEL_FILE}\n\n"
            "Please place the trained model in this path before running the app."
        )
        st.stop()

    loaded = joblib.load(MODEL_FILE)

    def _build_feature_matrix(input_df, feature_columns):
        feature_df = pd.DataFrame(0, index=input_df.index, columns=feature_columns)

        for col in ["tenure", "TotalCharges", "MonthlyCharges", "SeniorCitizen"]:
            if col in input_df.columns and col in feature_columns:
                feature_df[col] = input_df[col]

        if "gender" in input_df.columns and "gender" in feature_columns:
            feature_df["gender"] = input_df["gender"].map({"Male": 1, "Female": 0})

        if "Contract" in input_df.columns:
            for option in ["One year", "Two year", "Month-to-month"]:
                col_name = f"Contract_{option}"
                if col_name in feature_columns:
                    feature_df[col_name] = (input_df["Contract"] == option).astype(int)

        return feature_df

    if isinstance(loaded, dict) and "model" in loaded:
        base_model = loaded["model"]
        scaler = loaded.get("scaler")
        feature_columns = loaded.get("feature_columns", [])

        if not feature_columns:
            st.error("Loaded model is missing feature metadata.")
            st.stop()

        class CompatibleModel:
            def __init__(self, model, scaler, feature_columns):
                self.model = model
                self.scaler = scaler
                self.feature_columns = feature_columns

            def _prepare(self, input_df):
                matrix = _build_feature_matrix(input_df, self.feature_columns)
                if self.scaler is not None:
                    matrix = self.scaler.transform(matrix)
                return matrix

            def predict(self, input_df):
                return self.model.predict(self._prepare(input_df))

            def predict_proba(self, input_df):
                return self.model.predict_proba(self._prepare(input_df))

        model = CompatibleModel(base_model, scaler, feature_columns)

        model_df = df[["gender", "tenure", "Contract", "TotalCharges", "Churn"]].dropna().copy()
        model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})
        X = model_df[["gender", "tenure", "Contract", "TotalCharges"]]
        y = model_df["Churn"]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        return model, accuracy, auc, "Loaded compatible saved model"

    if not hasattr(loaded, "predict") or not hasattr(loaded, "predict_proba"):
        st.error("Loaded model is not compatible with this app.")
        st.stop()

    model_df = df[["gender", "tenure", "Contract", "TotalCharges", "Churn"]].dropna().copy()
    model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

    X = model_df[["gender", "tenure", "Contract", "TotalCharges"]]
    y = model_df["Churn"]

    y_pred = loaded.predict(X)
    y_prob = loaded.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    return loaded, accuracy, auc, "Loaded trained model"


df = load_data(DATA_FILE)
model, accuracy, auc, model_status = load_or_train_model(df)

# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Machine Learning Web Application</div>
    <div class="hero-title">📊 Customer Churn Prediction Dashboard</div>
    <div class="hero-subtitle">
        A clean and modern web application for predicting whether a telecom customer is likely to stay or churn.
        This final version uses the real Telco Customer Churn dataset, a machine learning model, and interactive
        EDA charts to support clear analysis and presentation.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- OVERVIEW --------------------
st.markdown('<div class="section-heading">Project Overview</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="card">
    <div class="muted">
        <b>Project Title:</b> Telco Customer Churn Prediction Web Application<br><br>
        <b>Dataset:</b> {DATA_FILE}<br><br>
        <b>Model:</b> Logistic Regression Pipeline<br><br>
        <b>Model Status:</b> {model_status}<br><br>
        <b>Accuracy:</b> {accuracy:.2%}<br><br>
        <b>ROC AUC:</b> {auc:.2%}<br><br>
        <b>Note:</b> Age is displayed in the interface for design purposes, but it is not used in the prediction
        because it does not exist in the original Telco dataset.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## 📋 Customer Form")
st.sidebar.caption("Fill in the customer details below to generate a churn prediction.")

gender = st.sidebar.radio(
    "1. What is your gender?",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "2. What is the age of the customer?",
    min_value=0,
    max_value=120,
    value=30,
    step=1,
    help="Age is shown in the UI only."
)

contract_ui = st.sidebar.selectbox(
    "3. What is the contract type?",
    ["Month-to-month", "One year", "Two years"]
)

total_charges = st.sidebar.number_input(
    "4. What is the total amount of charges for the customer?",
    min_value=0.0,
    value=1000.0,
    step=10.0,
    help="Please enter the total amount in dollars."
)

tenure = st.sidebar.number_input(
    "5. How many months has the customer been with the company?",
    min_value=0,
    max_value=120,
    value=12,
    step=1,
    help="Please enter the number of months."
)

st.sidebar.markdown("""
<div class="note">
This question-based form is designed to keep the interface simple, clean, and user-friendly.
</div>
""", unsafe_allow_html=True)

predict_btn = st.sidebar.button("🔍 Predict Churn")

contract_map = {
    "Month-to-month": "Month-to-month",
    "One year": "One year",
    "Two years": "Two year"
}
contract_model = contract_map[contract_ui]

# -------------------- INPUT / PREDICTION --------------------
input_df = pd.DataFrame([{
    "gender": gender,
    "tenure": tenure,
    "Contract": contract_model,
    "TotalCharges": total_charges
}])

pred_class = model.predict(input_df)[0]
pred_prob = model.predict_proba(input_df)[0][1]
prediction = "Likely to Churn" if pred_class == 1 else "Likely to Stay"

# -------------------- KPI ROW --------------------
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Customer Age</div>
        <div class="metric-value">{age} years</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Charges</div>
        <div class="metric-value">${total_charges:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Customer Tenure</div>
        <div class="metric-value">{tenure} months</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- MAIN PANELS --------------------
left_col, right_col = st.columns([1, 1.25], gap="large")

with left_col:
    st.markdown('<div class="section-heading">Customer Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="summary-grid">
            <b>Gender:</b> {gender}<br>
            <b>Age:</b> {age} years<br>
            <b>Contract Type:</b> {contract_ui}<br>
            <b>Total Charges:</b> ${total_charges:,.2f}<br>
            <b>Tenure:</b> {tenure} months
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Interface Description</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="muted">
            This section summarizes the entered customer information so the user can review all values clearly
            before interpreting the final machine learning prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        if prediction == "Likely to Churn":
            st.markdown(f"""
            <div class="card result-risk">
                <div class="result-title-risk">⚠️ {prediction}</div>
                <div class="muted">
                    Based on the trained model, this customer has a higher probability of leaving the service.
                    This can help businesses identify customers who may need retention strategies.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card result-good">
                <div class="result-title-good">✅ {prediction}</div>
                <div class="muted">
                    Based on the trained model, this customer appears more likely to stay with the service.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
            <div class="muted">
                Click <b>Predict Churn</b> in the sidebar to generate the final prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Churn Probability")
    st.progress(float(pred_prob))
    st.write(f"**Predicted Probability of Churn:** {pred_prob:.0%}")

    st.markdown("""
    <div class="muted">
        This probability bar provides a quick visual summary of churn risk, making the output easier to understand.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------- EDA --------------------
st.markdown('<div class="section-heading">Customer Insights Dashboard</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]

    fig1 = px.pie(
        churn_counts,
        values="Count",
        names="Churn",
        hole=0.6,
        title="Customer Churn Distribution"
    )
    fig1.update_layout(template="plotly_white", height=360, title_x=0.08)
    st.plotly_chart(fig1, width="stretch")

with col2:
    contract_churn = df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")

    fig2 = px.bar(
        contract_churn,
        x="Contract",
        y="Count",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )
    fig2.update_layout(
        template="plotly_white",
        height=360,
        title_x=0.05,
        xaxis_title="",
        yaxis_title="Customers"
    )
    st.plotly_chart(fig2, width="stretch")

col3, col4 = st.columns(2, gap="large")

with col3:
    fig3 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn"
    )
    fig3.update_layout(
        template="plotly_white",
        height=360,
        title_x=0.05,
        xaxis_title="",
        yaxis_title="Monthly Charges"
    )
    st.plotly_chart(fig3, width="stretch")

with col4:
    fig4 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        barmode="overlay",
        title="Tenure Distribution by Churn"
    )
    fig4.update_layout(
        template="plotly_white",
        height=360,
        title_x=0.05,
        xaxis_title="Tenure (Months)",
        yaxis_title="Count"
    )
    st.plotly_chart(fig4, width="stretch")

st.markdown("""
<div class="card">
    <div class="muted">
        <b>Key Insights:</b><br><br>
        • Customers with <b>month-to-month contracts</b> tend to churn more often.<br>
        • Higher <b>monthly charges</b> are associated with a greater churn risk.<br>
        • Customers with <b>shorter tenure</b> are more likely to leave.<br>
        • These insights support the machine learning prediction and make the dashboard more informative.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- FEEDBACK --------------------
st.markdown('<div class="section-heading">User Feedback</div>', unsafe_allow_html=True)
f1, f2 = st.columns([1, 1.2], gap="large")

with f1:
    feedback = st.radio("Was this prediction useful?", ["Yes", "No"], horizontal=True)
    comment = st.text_area("Additional comments", placeholder="Write your feedback here...")

with f2:
    st.markdown("""
    <div class="card">
        <div class="muted">
            This section improves interactivity and makes the application feel more complete and presentation-ready.
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer-note">
Final version: real dataset, real machine learning model, clean UI, and interactive EDA dashboard.
</div>
""", unsafe_allow_html=True)