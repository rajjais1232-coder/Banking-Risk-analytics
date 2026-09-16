"""
app.py
======
Banking Customer & Risk Analytics Dashboard
Professional Streamlit application with 8 analytical pages.

Usage:
    python -m streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Customer & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "banking_customers_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.csv")

# ── Theme colours ─────────────────────────────────────────────────────────────
COLOR_HIGH = "#e74c3c"
COLOR_MED  = "#f39c12"
COLOR_LOW  = "#27ae60"
COLOR_ACC  = "#3b82d4"
RISK_COLORS = {"High": COLOR_HIGH, "Medium": COLOR_MED, "Low": COLOR_LOW}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1f2328; }
    .metric-sub   { font-size: 12px; color: #57606a; margin-top: 2px; }
    .risk-high { color: #e74c3c; font-weight: 700; }
    .risk-med  { color: #f39c12; font-weight: 700; }
    .risk-low  { color: #27ae60; font-weight: 700; }
    .section-header {
        font-size: 20px; font-weight: 600; color: #1f2328;
        border-bottom: 2px solid #3b82d4;
        padding-bottom: 6px; margin: 20px 0 14px 0;
    }
    .insight-box {
        background: #f0f6ff; border-left: 4px solid #3b82d4;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
        font-size: 14px; color: #1f2328;
    }
    .warning-box {
        background: #fff8e1; border-left: 4px solid #f39c12;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
        font-size: 14px; color: #1f2328;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None, "Dataset not found. Please run:  python generate_data.py"
    try:
        df = pd.read_csv(DATA_PATH)
        if "Last_Transaction_Date" in df.columns:
            df["Last_Transaction_Date"] = pd.to_datetime(df["Last_Transaction_Date"], errors="coerce")
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model not found. Please run:  python generate_data.py"
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact["model"], artifact.get("feature_names", []), None
    except Exception as e:
        return None, None, str(e)


def kpi_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def fmt_inr(val):
    if val >= 1e7:
        return f"₹{val/1e7:.1f} Cr"
    if val >= 1e5:
        return f"₹{val/1e5:.1f} L"
    return f"₹{val:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Banking Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Executive Overview",
            "👥 Customer Analytics",
            "💳 Transaction Analytics",
            "🏦 Loan Analytics",
            "⚠️ Risk Analytics",
            "📊 Customer Segmentation",
            "🤖 ML Risk Prediction",
            "💡 Business Insights",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#57606a'>Banking Customer & Risk Analytics<br>"
        "Synthetic data — For portfolio demonstration only.</small>",
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df_full, data_err = load_data()
model, model_features, model_err = load_model()

if data_err:
    st.error(f"⚠️ {data_err}")
    st.stop()

df = df_full.copy()


# ── Segment helper ────────────────────────────────────────────────────────────
def assign_segment(row):
    if row["Account_Balance"] >= 500_000 and row["Credit_Score"] >= 750 and row["Number_of_Products"] >= 3:
        return "Premium"
    elif row["Account_Balance"] >= 200_000 and row["Credit_Score"] >= 650:
        return "High Value"
    elif row.get("Online_Banking_Usage", 0) == 1 and row.get("Mobile_Banking_Usage", 0) == 1:
        return "Digital First"
    elif row.get("Loan_Amount", 0) > 0 and row.get("Debt_to_Income_Ratio", 0) > 0.35:
        return "Loan Dependent"
    elif row["Churn_Status"] == 1:
        return "At Risk"
    elif row["Transaction_Count"] < 5:
        return "Low Engagement"
    else:
        return "Regular"


if "Segment" not in df.columns:
    df["Segment"] = df.apply(assign_segment, axis=1)


def credit_cat(score):
    if score < 500: return "Very Poor"
    if score < 600: return "Poor"
    if score < 700: return "Fair"
    if score < 750: return "Good"
    return "Excellent"


if "Credit_Category" not in df.columns:
    df["Credit_Category"] = df["Credit_Score"].apply(credit_cat)


def age_group(age):
    if age <= 25: return "18-25"
    if age <= 35: return "26-35"
    if age <= 45: return "36-45"
    if age <= 55: return "46-55"
    return "56+"


if "Age_Group" not in df.columns:
    df["Age_Group"] = df["Age"].apply(age_group)


# ==============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==============================================================================
if page == "🏠 Executive Overview":
    st.markdown("# 🏦 Banking Customer & Risk Analytics")
    st.markdown("### Executive Overview Dashboard")
    st.markdown("---")

    # KPI Row 1
    total = len(df)
    active = int((df["Churn_Status"] == 0).sum())
    total_dep = df["Account_Balance"].sum()
    total_loans = df["Loan_Amount"].sum() if "Loan_Amount" in df.columns else 0
    avg_cs = df["Credit_Score"].mean()
    default_rate = df["Default_Status"].mean() * 100 if "Default_Status" in df.columns else 0
    churn_rate = df["Churn_Status"].mean() * 100
    high_risk = int((df["Risk_Label"] == "High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Customers", f"{total:,}", f"Active: {active:,}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Deposits", fmt_inr(total_dep)), unsafe_allow_html=True)
    c3.markdown(kpi_card("Total Loans", fmt_inr(total_loans)), unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg Credit Score", f"{avg_cs:.0f}"), unsafe_allow_html=True)

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(kpi_card("Loan Default Rate", f"{default_rate:.1f}%"), unsafe_allow_html=True)
    c6.markdown(kpi_card("Customer Churn Rate", f"{churn_rate:.1f}%"), unsafe_allow_html=True)
    c7.markdown(kpi_card("High-Risk Customers", f"{high_risk:,}",
                         f"{high_risk/total*100:.1f}% of total"), unsafe_allow_html=True)
    c8.markdown(kpi_card("Avg Monthly Income", fmt_inr(df["Monthly_Income"].mean())), unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_cnt = df["Risk_Label"].value_counts().reset_index()
        risk_cnt.columns = ["Risk", "Count"]
        fig = px.pie(risk_cnt, values="Count", names="Risk",
                     color="Risk", color_discrete_map=RISK_COLORS,
                     hole=0.45)
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Customers by Account Type</div>', unsafe_allow_html=True)
        acc_cnt = df["Account_Type"].value_counts().reset_index()
        acc_cnt.columns = ["Account_Type", "Count"]
        fig2 = px.bar(acc_cnt, x="Account_Type", y="Count",
                      color="Account_Type", color_discrete_sequence=px.colors.qualitative.Set2,
                      text="Count")
        fig2.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Top 10 States by Customers</div>', unsafe_allow_html=True)
        state_cnt = df["State"].value_counts().head(10).reset_index()
        state_cnt.columns = ["State", "Count"]
        fig3 = px.bar(state_cnt, x="Count", y="State", orientation="h",
                      color="Count", color_continuous_scale="Blues", text="Count")
        fig3.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False,
                           coloraxis_showscale=False)
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Credit Score Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(df, x="Credit_Score", nbins=40,
                            color_discrete_sequence=[COLOR_ACC])
        fig4.update_layout(height=340, margin=dict(t=10, b=10),
                           xaxis_title="Credit Score", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    # Risk by state
    st.markdown('<div class="section-header">Risk Distribution by State</div>', unsafe_allow_html=True)
    state_risk = df.groupby(["State", "Risk_Label"]).size().reset_index(name="Count")
    fig5 = px.bar(state_risk, x="State", y="Count", color="Risk_Label",
                  color_discrete_map=RISK_COLORS, barmode="stack")
    fig5.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)


# ==============================================================================
# PAGE 2 — CUSTOMER ANALYTICS
# ==============================================================================
elif page == "👥 Customer Analytics":
    st.markdown("## 👥 Customer Analytics")
    st.markdown("---")

    # Filters
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        age_range = fc1.slider("Age Range", 18, 75, (18, 75))
        genders = fc2.multiselect("Gender", df["Gender"].unique().tolist(), default=df["Gender"].unique().tolist())
        states_sel = fc3.multiselect("State", sorted(df["State"].unique()), default=sorted(df["State"].unique()))
        accs = fc4.multiselect("Account Type", df["Account_Type"].unique().tolist(), default=df["Account_Type"].unique().tolist())
        cs_range = fc5.slider("Credit Score", 300, 900, (300, 900))

    filt = df[
        df["Age"].between(age_range[0], age_range[1]) &
        df["Gender"].isin(genders) &
        df["State"].isin(states_sel) &
        df["Account_Type"].isin(accs) &
        df["Credit_Score"].between(cs_range[0], cs_range[1])
    ]
    st.caption(f"Showing {len(filt):,} customers")

    if filt.empty:
        st.warning("No customers match the selected filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Age Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(filt, x="Age", nbins=30, color_discrete_sequence=[COLOR_ACC])
            fig.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Gender Distribution</div>', unsafe_allow_html=True)
            g_cnt = filt["Gender"].value_counts().reset_index()
            g_cnt.columns = ["Gender", "Count"]
            fig2 = px.pie(g_cnt, values="Count", names="Gender",
                          color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
            fig2.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-header">Monthly Income Distribution</div>', unsafe_allow_html=True)
            fig3 = px.histogram(filt, x="Monthly_Income", nbins=40, color_discrete_sequence=["#7c5cd8"])
            fig3.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">Account Balance Distribution</div>', unsafe_allow_html=True)
            fig4 = px.histogram(filt, x="Account_Balance", nbins=40, color_discrete_sequence=["#27ae60"])
            fig4.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<div class="section-header">Average Income by Occupation</div>', unsafe_allow_html=True)
            occ_inc = filt.groupby("Occupation")["Monthly_Income"].mean().sort_values(ascending=False).reset_index()
            fig5 = px.bar(occ_inc, x="Monthly_Income", y="Occupation", orientation="h",
                          color="Monthly_Income", color_continuous_scale="Blues")
            fig5.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            st.markdown('<div class="section-header">Tenure Distribution</div>', unsafe_allow_html=True)
            fig6 = px.histogram(filt, x="Tenure_Years", nbins=20, color_discrete_sequence=["#e67e22"])
            fig6.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="section-header">Products per Customer</div>', unsafe_allow_html=True)
        prod_cnt = filt["Number_of_Products"].value_counts().sort_index().reset_index()
        prod_cnt.columns = ["Products", "Count"]
        fig7 = px.bar(prod_cnt, x="Products", y="Count", text="Count",
                      color_discrete_sequence=[COLOR_ACC])
        fig7.update_layout(height=280, margin=dict(t=10))
        fig7.update_traces(textposition="outside")
        st.plotly_chart(fig7, use_container_width=True)


# ==============================================================================
# PAGE 3 — TRANSACTION ANALYTICS
# ==============================================================================
elif page == "💳 Transaction Analytics":
    st.markdown("## 💳 Transaction Analytics")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Total Transactions", f"{df['Transaction_Count'].sum():,}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Txn Volume", fmt_inr(df["Transaction_Amount"].sum())), unsafe_allow_html=True)
    c3.markdown(kpi_card("Avg Txn Value", fmt_inr(df["Average_Transaction_Value"].mean())), unsafe_allow_html=True)
    c4.markdown(kpi_card("Online Banking Users", f"{df['Online_Banking_Usage'].sum():,}",
                         f"{df['Online_Banking_Usage'].mean()*100:.0f}%"), unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Transaction_Amount", nbins=40, color_discrete_sequence=[COLOR_ACC])
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Transaction Count Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="Transaction_Count", nbins=30, color_discrete_sequence=["#7c5cd8"])
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Digital Banking Adoption</div>', unsafe_allow_html=True)
        digital_data = pd.DataFrame({
            "Channel": ["Online Banking", "Mobile Banking", "Both", "Neither"],
            "Users": [
                int(df["Online_Banking_Usage"].sum()),
                int(df["Mobile_Banking_Usage"].sum()),
                int(((df["Online_Banking_Usage"] == 1) & (df["Mobile_Banking_Usage"] == 1)).sum()),
                int(((df["Online_Banking_Usage"] == 0) & (df["Mobile_Banking_Usage"] == 0)).sum()),
            ]
        })
        fig3 = px.bar(digital_data, x="Channel", y="Users", text="Users",
                      color="Channel", color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=300, margin=dict(t=10), showlegend=False)
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">ATM Usage Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(df, x="ATM_Usage", nbins=20, color_discrete_sequence=["#e67e22"])
        fig4.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Average Transaction Amount by Occupation</div>', unsafe_allow_html=True)
    occ_txn = df.groupby("Occupation")["Transaction_Amount"].mean().sort_values(ascending=False).reset_index()
    fig5 = px.bar(occ_txn, x="Occupation", y="Transaction_Amount",
                  color="Transaction_Amount", color_continuous_scale="Blues",
                  text=occ_txn["Transaction_Amount"].apply(lambda v: fmt_inr(v)))
    fig5.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
    fig5.update_traces(textposition="outside")
    st.plotly_chart(fig5, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="section-header">Transaction Amount vs Risk Label</div>', unsafe_allow_html=True)
        risk_txn = df.groupby("Risk_Label")["Transaction_Amount"].mean().reset_index()
        fig6 = px.bar(risk_txn, x="Risk_Label", y="Transaction_Amount",
                      color="Risk_Label", color_discrete_map=RISK_COLORS,
                      text=risk_txn["Transaction_Amount"].apply(lambda v: fmt_inr(v)))
        fig6.update_layout(height=280, margin=dict(t=10), showlegend=False)
        fig6.update_traces(textposition="outside")
        st.plotly_chart(fig6, use_container_width=True)

    with col6:
        st.markdown('<div class="section-header">Branch Visits Distribution</div>', unsafe_allow_html=True)
        fig7 = px.histogram(df, x="Branch_Visits", nbins=10, color_discrete_sequence=["#16a085"])
        fig7.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig7, use_container_width=True)


# ==============================================================================
# PAGE 4 — LOAN ANALYTICS
# ==============================================================================
elif page == "🏦 Loan Analytics":
    st.markdown("## 🏦 Loan Analytics")
    st.markdown("---")

    loan_df = df[df["Loan_Amount"] > 0].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Loan Customers", f"{len(loan_df):,}",
                         f"{len(loan_df)/len(df)*100:.0f}% of total"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Loan Portfolio", fmt_inr(loan_df["Loan_Amount"].sum())), unsafe_allow_html=True)
    c3.markdown(kpi_card("Avg Loan Amount", fmt_inr(loan_df["Loan_Amount"].mean())), unsafe_allow_html=True)
    default_r = loan_df["Default_Status"].mean() * 100 if "Default_Status" in loan_df.columns else 0
    c4.markdown(kpi_card("Loan Default Rate", f"{default_r:.1f}%"), unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Loan Status Breakdown</div>', unsafe_allow_html=True)
        ls_cnt = loan_df["Loan_Status"].value_counts().reset_index()
        ls_cnt.columns = ["Status", "Count"]
        fig = px.pie(ls_cnt, values="Count", names="Status",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Loan Amount Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(loan_df, x="Loan_Amount", nbins=40, color_discrete_sequence=[COLOR_ACC])
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Debt-to-Income Ratio Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(loan_df, x="Debt_to_Income_Ratio", nbins=40, color_discrete_sequence=[COLOR_HIGH])
        fig3.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Total Loans by State (Top 10)</div>', unsafe_allow_html=True)
        state_loans = loan_df.groupby("State")["Loan_Amount"].sum().sort_values(ascending=False).head(10).reset_index()
        state_loans["Loan_Cr"] = state_loans["Loan_Amount"] / 1e7
        fig4 = px.bar(state_loans, x="Loan_Cr", y="State", orientation="h",
                      color="Loan_Cr", color_continuous_scale="Reds",
                      text=state_loans["Loan_Cr"].apply(lambda v: f"₹{v:.1f}Cr"))
        fig4.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Missed Payments vs Default Rate</div>', unsafe_allow_html=True)
    loan_df["Missed_Bucket"] = pd.cut(loan_df["Missed_Payments"],
                                       bins=[-1, 0, 2, 5, 100],
                                       labels=["0", "1-2", "3-5", "6+"])
    mp_def = loan_df.groupby("Missed_Bucket", observed=True).agg(
        Count=("Default_Status", "count"),
        Default_Rate=("Default_Status", "mean")
    ).reset_index()
    mp_def["Default_Rate_Pct"] = mp_def["Default_Rate"] * 100

    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Bar(x=mp_def["Missed_Bucket"], y=mp_def["Count"],
                          name="Customers", marker_color=COLOR_ACC), secondary_y=False)
    fig5.add_trace(go.Scatter(x=mp_def["Missed_Bucket"], y=mp_def["Default_Rate_Pct"],
                              name="Default Rate %", mode="lines+markers",
                              line=dict(color=COLOR_HIGH, width=2)), secondary_y=True)
    fig5.update_layout(height=320, margin=dict(t=10))
    fig5.update_yaxes(title_text="Customer Count", secondary_y=False)
    fig5.update_yaxes(title_text="Default Rate (%)", secondary_y=True)
    st.plotly_chart(fig5, use_container_width=True)


# ==============================================================================
# PAGE 5 — RISK ANALYTICS
# ==============================================================================
elif page == "⚠️ Risk Analytics":
    st.markdown("## ⚠️ Risk Analytics")
    st.markdown("---")

    risk_filter = st.selectbox("Filter by Risk Category", ["All", "High", "Medium", "Low"])
    rdf = df if risk_filter == "All" else df[df["Risk_Label"] == risk_filter]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("High Risk", f"{(df['Risk_Label']=='High').sum():,}",
                         f"{(df['Risk_Label']=='High').mean()*100:.1f}%"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Medium Risk", f"{(df['Risk_Label']=='Medium').sum():,}",
                         f"{(df['Risk_Label']=='Medium').mean()*100:.1f}%"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Low Risk", f"{(df['Risk_Label']=='Low').sum():,}",
                         f"{(df['Risk_Label']=='Low').mean()*100:.1f}%"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Avg DTI (High Risk)",
                         f"{df[df['Risk_Label']=='High']['Debt_to_Income_Ratio'].mean():.3f}"), unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Risk by Age Group</div>', unsafe_allow_html=True)
        age_risk = rdf.groupby(["Age_Group", "Risk_Label"]).size().reset_index(name="Count")
        fig = px.bar(age_risk, x="Age_Group", y="Count", color="Risk_Label",
                     color_discrete_map=RISK_COLORS, barmode="group")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Risk by Credit Score Category</div>', unsafe_allow_html=True)
        cc_risk = rdf.groupby(["Credit_Category", "Risk_Label"]).size().reset_index(name="Count")
        cc_order = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]
        cc_risk["Credit_Category"] = pd.Categorical(cc_risk["Credit_Category"], categories=cc_order, ordered=True)
        cc_risk = cc_risk.sort_values("Credit_Category")
        fig2 = px.bar(cc_risk, x="Credit_Category", y="Count", color="Risk_Label",
                      color_discrete_map=RISK_COLORS, barmode="stack")
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Risk by Missed Payments</div>', unsafe_allow_html=True)
        rdf2 = rdf.copy()
        rdf2["Missed_Bucket"] = pd.cut(rdf2["Missed_Payments"], bins=[-1,0,2,5,100], labels=["0","1-2","3-5","6+"])
        mp_risk = rdf2.groupby(["Missed_Bucket","Risk_Label"], observed=True).size().reset_index(name="Count")
        fig3 = px.bar(mp_risk, x="Missed_Bucket", y="Count", color="Risk_Label",
                      color_discrete_map=RISK_COLORS, barmode="stack")
        fig3.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Risk by Debt-to-Income Ratio</div>', unsafe_allow_html=True)
        fig4 = px.violin(rdf, y="Debt_to_Income_Ratio", x="Risk_Label",
                         color="Risk_Label", color_discrete_map=RISK_COLORS, box=True)
        fig4.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Risk heatmap by state and risk
    st.markdown('<div class="section-header">Risk Heatmap — State × Credit Category</div>', unsafe_allow_html=True)
    heat_data = rdf.groupby(["State", "Credit_Category"]).apply(
        lambda x: (x["Risk_Label"] == "High").mean() * 100
    ).reset_index(name="High_Risk_Pct")
    heat_pivot = heat_data.pivot(index="State", columns="Credit_Category", values="High_Risk_Pct").fillna(0)
    fig5 = px.imshow(heat_pivot, color_continuous_scale="Reds",
                     labels=dict(color="High Risk %"),
                     aspect="auto")
    fig5.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">High-Risk Customer Sample</div>', unsafe_allow_html=True)
    high_df = df[df["Risk_Label"] == "High"][
        ["Customer_ID", "Age", "Monthly_Income", "Credit_Score",
         "Missed_Payments", "Debt_to_Income_Ratio", "Default_Status", "State"]
    ].head(20)
    st.dataframe(high_df, use_container_width=True)


# ==============================================================================
# PAGE 6 — CUSTOMER SEGMENTATION
# ==============================================================================
elif page == "📊 Customer Segmentation":
    st.markdown("## 📊 Customer Segmentation")
    st.markdown("---")
    st.markdown(
        '<div class="insight-box">Customers are segmented using business rules based on account balance, '
        'credit score, digital engagement, loan dependence, and churn behaviour.</div>',
        unsafe_allow_html=True
    )

    seg_summary = df.groupby("Segment").agg(
        Count=("Customer_ID", "count"),
        Avg_Income=("Monthly_Income", "mean"),
        Avg_Balance=("Account_Balance", "mean"),
        Avg_Credit_Score=("Credit_Score", "mean"),
        Avg_Txn_Amount=("Transaction_Amount", "mean"),
        Churn_Rate=("Churn_Status", "mean"),
        High_Risk_Pct=("Risk_Label", lambda x: (x == "High").mean() * 100),
    ).reset_index()
    seg_summary = seg_summary.sort_values("Count", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Segment Size</div>', unsafe_allow_html=True)
        fig = px.pie(seg_summary, values="Count", names="Segment",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Average Income by Segment</div>', unsafe_allow_html=True)
        fig2 = px.bar(seg_summary.sort_values("Avg_Income", ascending=True),
                      x="Avg_Income", y="Segment", orientation="h",
                      color="Avg_Income", color_continuous_scale="Blues",
                      text=seg_summary.sort_values("Avg_Income", ascending=True)["Avg_Income"].apply(fmt_inr))
        fig2.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Average Balance by Segment</div>', unsafe_allow_html=True)
        fig3 = px.bar(seg_summary.sort_values("Avg_Balance", ascending=False),
                      x="Segment", y="Avg_Balance",
                      color="Avg_Balance", color_continuous_scale="Greens",
                      text=seg_summary.sort_values("Avg_Balance", ascending=False)["Avg_Balance"].apply(fmt_inr))
        fig3.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">High-Risk % by Segment</div>', unsafe_allow_html=True)
        fig4 = px.bar(seg_summary.sort_values("High_Risk_Pct", ascending=False),
                      x="Segment", y="High_Risk_Pct",
                      color="High_Risk_Pct", color_continuous_scale="Reds", text_auto=".1f")
        fig4.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Segment Summary Table</div>', unsafe_allow_html=True)
    disp = seg_summary.copy()
    disp["Avg_Income"] = disp["Avg_Income"].apply(lambda v: f"₹{v:,.0f}")
    disp["Avg_Balance"] = disp["Avg_Balance"].apply(lambda v: f"₹{v:,.0f}")
    disp["Avg_Txn_Amount"] = disp["Avg_Txn_Amount"].apply(lambda v: f"₹{v:,.0f}")
    disp["Churn_Rate"] = disp["Churn_Rate"].apply(lambda v: f"{v*100:.1f}%")
    disp["High_Risk_Pct"] = disp["High_Risk_Pct"].apply(lambda v: f"{v:.1f}%")
    disp["Avg_Credit_Score"] = disp["Avg_Credit_Score"].apply(lambda v: f"{v:.0f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Risk by segment
    st.markdown('<div class="section-header">Risk Distribution by Segment</div>', unsafe_allow_html=True)
    seg_risk = df.groupby(["Segment", "Risk_Label"]).size().reset_index(name="Count")
    fig5 = px.bar(seg_risk, x="Segment", y="Count", color="Risk_Label",
                  color_discrete_map=RISK_COLORS, barmode="stack")
    fig5.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig5, use_container_width=True)


# ==============================================================================
# PAGE 7 — ML RISK PREDICTION
# ==============================================================================
elif page == "🤖 ML Risk Prediction":
    st.markdown("## 🤖 ML Risk Prediction")
    st.markdown("---")
    st.markdown(
        '<div class="warning-box">⚠️ This is an <strong>analytical model</strong> built on synthetic data '
        'for portfolio demonstration purposes only. It is NOT a real-world credit approval or '
        'financial advisory tool. Results should not be used for any actual financial decisions.</div>',
        unsafe_allow_html=True
    )

    if model_err:
        st.error(f"Model not loaded: {model_err}")
        st.info("Run  `python generate_data.py`  to train and save the model.")
        st.stop()

    # Show model metrics
    if os.path.exists(METRICS_PATH):
        with st.expander("📋 Model Comparison", expanded=False):
            m_df = pd.read_csv(METRICS_PATH)
            st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown("### Enter Customer Information")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_age = st.number_input("Age", 18, 80, 35)
            p_income = st.number_input("Monthly Income (₹)", 5000, 1000000, 45000, step=1000)
            p_credit = st.number_input("Credit Score", 300, 900, 680)
            p_balance = st.number_input("Account Balance (₹)", 0, 5000000, 150000, step=5000)
            p_tenure = st.number_input("Tenure Years", 0, 30, 4)
        with col2:
            p_loan = st.number_input("Loan Amount (₹)", 0, 5000000, 0, step=10000)
            p_emi = st.number_input("EMI (₹/month)", 0, 200000, 0, step=500)
            p_expenses = st.number_input("Monthly Expenses (₹)", 0, 500000, 20000, step=1000)
            p_missed = st.slider("Missed Payments", 0, 15, 0)
            p_late = st.slider("Late Payments", 0, 15, 0)
        with col3:
            p_txn_count = st.number_input("Transaction Count/Month", 1, 120, 15)
            p_txn_amount = st.number_input("Transaction Amount (₹)", 0, 2000000, 30000, step=1000)
            p_products = st.slider("Number of Products", 1, 5, 2)
            p_complaints = st.slider("Complaints", 0, 10, 0)
            p_satisfaction = st.slider("Customer Satisfaction", 1.0, 5.0, 3.5, 0.1)

        col4, col5 = st.columns(2)
        with col4:
            p_existing_loans = st.slider("Existing Loans", 0, 5, 0)
            p_online = st.selectbox("Online Banking", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            p_mobile = st.selectbox("Mobile Banking", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        with col5:
            p_atm = st.slider("ATM Usage/Month", 0, 20, 3)
            p_branch = st.slider("Branch Visits/Month", 0, 10, 1)
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            p_occ = st.selectbox("Occupation",
                                  ["Salaried", "Self-Employed", "Business Owner",
                                   "Professional", "Retired", "Freelancer", "Homemaker", "Student"])
            p_acc = st.selectbox("Account Type", ["Savings", "Current", "Salary", "NRI"])

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        from sklearn.preprocessing import LabelEncoder
        from src.feature_engineering import FEATURE_COLS

        dti = round(p_emi / max(p_income, 1), 3) if p_emi > 0 else 0.0

        gender_le = LabelEncoder().fit(["Female", "Male", "Other"])
        occ_le = LabelEncoder().fit(sorted(["Salaried","Self-Employed","Business Owner",
                                            "Professional","Retired","Freelancer","Homemaker","Student"]))
        acc_le = LabelEncoder().fit(["Current","NRI","Salary","Savings"])

        avg_txn = p_txn_amount // max(p_txn_count, 1)

        input_dict = {
            "Age": p_age, "Monthly_Income": p_income, "Credit_Score": p_credit,
            "Account_Balance": p_balance, "Tenure_Years": p_tenure,
            "Number_of_Products": p_products, "Loan_Amount": p_loan,
            "EMI": p_emi, "Monthly_Expenses": p_expenses,
            "Transaction_Count": p_txn_count, "Transaction_Amount": p_txn_amount,
            "Average_Transaction_Value": avg_txn,
            "Missed_Payments": p_missed, "Late_Payments": p_late,
            "Existing_Loans": p_existing_loans, "Debt_to_Income_Ratio": dti,
            "Complaints": p_complaints, "Support_Calls": p_complaints + 1,
            "ATM_Usage": p_atm, "Branch_Visits": p_branch,
            "Online_Banking_Usage": p_online, "Mobile_Banking_Usage": p_mobile,
            "Credit_Card_Usage": 1, "Debit_Card_Usage": 1,
            "Customer_Satisfaction": p_satisfaction, "Days_Since_Transaction": 7,
            "Gender_enc": int(gender_le.transform([p_gender])[0]),
            "Occupation_enc": int(occ_le.transform([p_occ])[0]),
            "Account_Type_enc": int(acc_le.transform([p_acc])[0]),
        }

        available_feats = [c for c in FEATURE_COLS if c in input_dict]
        X_input = np.array([[input_dict[c] for c in available_feats]])

        try:
            proba = model.predict_proba(X_input)[0]
            pred_class = model.predict(X_input)[0]
            label_map = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}
            risk_label = label_map.get(pred_class, "UNKNOWN")
            risk_color = {0: COLOR_LOW, 1: COLOR_MED, 2: COLOR_HIGH}.get(pred_class, "#888")

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            rc1, rc2, rc3 = st.columns(3)
            rc1.markdown(
                f'<div class="metric-card"><div class="metric-label">Risk Category</div>'
                f'<div class="metric-value" style="color:{risk_color}">{risk_label}</div></div>',
                unsafe_allow_html=True
            )
            rc2.markdown(
                kpi_card("Risk Probability", f"{proba[pred_class]*100:.1f}%"),
                unsafe_allow_html=True
            )
            rc3.markdown(
                kpi_card("Debt-to-Income Ratio", f"{dti:.3f}",
                         "High risk if > 0.5"),
                unsafe_allow_html=True
            )

            st.markdown("")
            prob_df = pd.DataFrame({
                "Risk Category": ["Low Risk", "Medium Risk", "High Risk"],
                "Probability (%)": [round(p*100, 2) for p in proba]
            })
            fig_prob = px.bar(prob_df, x="Risk Category", y="Probability (%)",
                              color="Risk Category",
                              color_discrete_map={"Low Risk": COLOR_LOW,
                                                  "Medium Risk": COLOR_MED,
                                                  "High Risk": COLOR_HIGH},
                              text="Probability (%)")
            fig_prob.update_layout(height=280, margin=dict(t=10), showlegend=False)
            fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Feature importance
            try:
                clf = model.named_steps["clf"]
                if hasattr(clf, "feature_importances_"):
                    fi = pd.Series(clf.feature_importances_, index=available_feats)
                    fi_top = fi.sort_values(ascending=False).head(10).reset_index()
                    fi_top.columns = ["Feature", "Importance"]
                    st.markdown("### 📊 Top Contributing Factors")
                    fig_fi = px.bar(fi_top.sort_values("Importance"),
                                    x="Importance", y="Feature", orientation="h",
                                    color="Importance", color_continuous_scale="Blues")
                    fig_fi.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                pass

            # Key risk factors narrative
            st.markdown("### 🔍 Key Risk Indicators")
            indicators = []
            if p_credit < 600:
                indicators.append(f"⚠️ Credit score ({p_credit}) is below 600 — elevated credit risk.")
            if dti > 0.4:
                indicators.append(f"⚠️ Debt-to-Income ratio ({dti:.2f}) exceeds 0.4 — high financial stress.")
            if p_missed > 2:
                indicators.append(f"⚠️ {p_missed} missed payments — payment reliability concern.")
            if p_loan > p_income * 18:
                indicators.append(f"⚠️ Loan amount is {p_loan/p_income:.0f}x monthly income — high loan burden.")
            if p_existing_loans > 2:
                indicators.append(f"⚠️ {p_existing_loans} existing loans — over-leveraged.")
            if not indicators:
                indicators.append("✅ No major risk indicators detected in the provided data.")
            for ind in indicators:
                st.markdown(f'<div class="insight-box">{ind}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==============================================================================
# PAGE 8 — BUSINESS INSIGHTS
# ==============================================================================
elif page == "💡 Business Insights":
    st.markdown("## 💡 Business Insights & Recommendations")
    st.markdown("---")
    st.markdown(
        '<div class="insight-box">All insights below are <strong>calculated from the actual dataset</strong>. '
        'No conclusions are hard-coded.</div>',
        unsafe_allow_html=True
    )

    # ── Calculated insights ───────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Insights", "📋 KPI Dashboard", "📌 Recommendations"])

    with tab1:
        st.markdown("### Data-Driven Insights")

        # Insight 1: High risk segments
        seg_risk_pct = df.groupby("Segment").apply(
            lambda x: (x["Risk_Label"] == "High").mean() * 100
        ).sort_values(ascending=False)
        top_risk_seg = seg_risk_pct.index[0]
        st.markdown(
            f'<div class="insight-box">🔴 <strong>Riskiest Segment:</strong> '
            f'"{top_risk_seg}" customers have the highest proportion of High-Risk customers '
            f'({seg_risk_pct.iloc[0]:.1f}%).</div>',
            unsafe_allow_html=True
        )

        # Insight 2: State default rate
        state_def = df.groupby("State")["Default_Status"].mean().sort_values(ascending=False)
        top_state = state_def.index[0]
        st.markdown(
            f'<div class="insight-box">🗺️ <strong>Highest Default State:</strong> '
            f'{top_state} has the highest loan default rate ({state_def.iloc[0]*100:.1f}%).</div>',
            unsafe_allow_html=True
        )

        # Insight 3: Credit score risk correlation
        low_cs_risk = df[df["Credit_Score"] < 600]["Risk_Label"].eq("High").mean() * 100
        high_cs_risk = df[df["Credit_Score"] >= 700]["Risk_Label"].eq("High").mean() * 100
        st.markdown(
            f'<div class="insight-box">💳 <strong>Credit Score & Risk:</strong> '
            f'Customers with credit score below 600 have {low_cs_risk:.1f}% high-risk rate vs '
            f'{high_cs_risk:.1f}% for scores above 700.</div>',
            unsafe_allow_html=True
        )

        # Insight 4: DTI correlation
        high_dti_risk = df[df["Debt_to_Income_Ratio"] > 0.5]["Risk_Label"].eq("High").mean() * 100
        low_dti_risk  = df[df["Debt_to_Income_Ratio"] <= 0.3]["Risk_Label"].eq("High").mean() * 100
        st.markdown(
            f'<div class="insight-box">📈 <strong>Debt-to-Income Impact:</strong> '
            f'Customers with DTI > 0.5 show {high_dti_risk:.1f}% high-risk rate vs '
            f'{low_dti_risk:.1f}% for DTI ≤ 0.3.</div>',
            unsafe_allow_html=True
        )

        # Insight 5: Missed payments
        mp_risk = df.groupby(pd.cut(df["Missed_Payments"], bins=[-1,0,2,5,100],
                                     labels=["0","1-2","3-5","6+"])
                              , observed=True)["Risk_Label"].apply(
            lambda x: (x == "High").mean() * 100
        )
        st.markdown(
            f'<div class="insight-box">⚠️ <strong>Missed Payments:</strong> '
            f'Customers with 6+ missed payments have {mp_risk.get("6+", 0):.1f}% high-risk rate '
            f'vs {mp_risk.get("0", 0):.1f}% for those with no missed payments.</div>',
            unsafe_allow_html=True
        )

        # Insight 6: Churn by products
        prod_churn = df.groupby("Number_of_Products")["Churn_Status"].mean().sort_values(ascending=False)
        st.markdown(
            f'<div class="insight-box">📦 <strong>Product Cross-sell & Churn:</strong> '
            f'Customers with only 1 product have {prod_churn.get(1,0)*100:.1f}% churn rate vs '
            f'{prod_churn.get(5,0)*100:.1f}% for those with 5 products.</div>',
            unsafe_allow_html=True
        )

        # Insight 7: Account type balances
        acc_bal = df.groupby("Account_Type")["Account_Balance"].mean().sort_values(ascending=False)
        st.markdown(
            f'<div class="insight-box">🏦 <strong>Account Type Balances:</strong> '
            f'{acc_bal.index[0]} accounts hold the highest average balance '
            f'({fmt_inr(acc_bal.iloc[0])}).</div>',
            unsafe_allow_html=True
        )

        # Insight 8: Digital vs traditional high risk
        digital_risk = df[(df["Online_Banking_Usage"]==1)&(df["Mobile_Banking_Usage"]==1)]["Risk_Label"].eq("High").mean()*100
        trad_risk = df[(df["Online_Banking_Usage"]==0)&(df["Mobile_Banking_Usage"]==0)]["Risk_Label"].eq("High").mean()*100
        st.markdown(
            f'<div class="insight-box">📱 <strong>Digital Banking & Risk:</strong> '
            f'Fully digital customers have {digital_risk:.1f}% high-risk rate vs '
            f'{trad_risk:.1f}% for non-digital customers.</div>',
            unsafe_allow_html=True
        )

        # Charts
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Default Rate by State</div>', unsafe_allow_html=True)
            state_d = df.groupby("State")["Default_Status"].mean().mul(100).sort_values(ascending=False).reset_index()
            state_d.columns = ["State", "Default_Rate"]
            fig = px.bar(state_d, x="State", y="Default_Rate", color="Default_Rate",
                         color_continuous_scale="Reds", text_auto=".1f")
            fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Churn Rate by Segment</div>', unsafe_allow_html=True)
            seg_churn = df.groupby("Segment")["Churn_Status"].mean().mul(100).sort_values(ascending=False).reset_index()
            seg_churn.columns = ["Segment", "Churn_Rate"]
            fig2 = px.bar(seg_churn, x="Segment", y="Churn_Rate", color="Churn_Rate",
                          color_continuous_scale="Oranges", text_auto=".1f")
            fig2.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### KPI Dashboard")
        sys.path.insert(0, BASE_DIR)
        try:
            from src.analytics import calculate_kpis
            kpis = calculate_kpis(df)
            kpi_items = list(kpis.items())
            for i in range(0, len(kpi_items), 4):
                cols = st.columns(4)
                for j, (k, v) in enumerate(kpi_items[i:i+4]):
                    val_str = f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                    cols[j].markdown(kpi_card(k, val_str), unsafe_allow_html=True)
                st.markdown("")
        except Exception as e:
            st.error(f"KPI calculation failed: {e}")

    with tab3:
        st.markdown("### Business Recommendations")

        recs = []

        # Conditional recommendation 1
        high_risk_pct = (df["Risk_Label"] == "High").mean() * 100
        if high_risk_pct > 15:
            recs.append(
                f"🔴 <strong>Risk Mitigation Priority:</strong> {high_risk_pct:.1f}% of customers are "
                f"High Risk. Implement enhanced credit monitoring, early warning systems, and "
                f"proactive outreach for this group."
            )

        # Conditional recommendation 2
        churn_rate = df["Churn_Status"].mean() * 100
        if churn_rate > 15:
            recs.append(
                f"💔 <strong>Retention Programme:</strong> Customer churn rate is {churn_rate:.1f}%. "
                f"Focus retention on customers with 1 product, low satisfaction scores, "
                f"and >60 days since last transaction."
            )

        # Conditional recommendation 3
        avg_products = df["Number_of_Products"].mean()
        if avg_products < 2.5:
            recs.append(
                f"📦 <strong>Cross-Sell Opportunity:</strong> Average products per customer is "
                f"{avg_products:.2f}. Targeted cross-selling to single-product customers can "
                f"reduce churn and increase lifetime value."
            )

        # Conditional recommendation 4
        digital_pct = ((df["Online_Banking_Usage"]==1)|(df["Mobile_Banking_Usage"]==1)).mean()*100
        if digital_pct < 80:
            recs.append(
                f"📱 <strong>Digital Adoption:</strong> {digital_pct:.1f}% of customers use digital banking. "
                f"Incentivise digital onboarding — especially for branch-dependent customers — "
                f"to reduce operating costs and improve engagement."
            )

        # Conditional recommendation 5
        high_dti = (df["Debt_to_Income_Ratio"] > 0.5).mean() * 100
        if high_dti > 10:
            recs.append(
                f"📊 <strong>DTI Monitoring:</strong> {high_dti:.1f}% of customers have DTI > 0.5. "
                f"Implement automated alerts for customers approaching this threshold and "
                f"restrict further lending for those already above it."
            )

        # Conditional recommendation 6
        low_cs = (df["Credit_Score"] < 600).mean() * 100
        if low_cs > 15:
            recs.append(
                f"💳 <strong>Credit Health Programme:</strong> {low_cs:.1f}% of customers have "
                f"credit scores below 600. Offer financial literacy resources and structured "
                f"repayment plans to improve credit health."
            )

        if not recs:
            recs.append("✅ The portfolio appears to be in good health. Continue monitoring key risk indicators.")

        for rec in recs:
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)
