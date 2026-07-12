from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_USER = "fraud_user"
DB_PASSWORD = "fraud_pass"
DB_HOST = "127.0.0.1"
DB_PORT = "55432"
DB_NAME = "fraud_warehouse"

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

VECTOR_DIR = Path("rag_assistant/vector_store")
CASE_SUMMARY_DIR = Path("rag_assistant/data/case_summaries")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 90% 0%, rgba(45, 224, 145, 0.10), transparent 23%),
            #050505;
        color: #ffffff;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -2px;
        margin-top: 16px;
    }

    .hero-subtitle {
        font-size: 38px;
        font-weight: 300;
        line-height: 1.05;
        letter-spacing: -1.5px;
        color: #d5d5d5;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin: 5px 0 14px 0;
    }

    .panel {
        background: #181818;
        border: 1px solid #2e2e2e;
        border-radius: 22px;
        padding: 20px;
        margin-top: 10px;
    }

    .metric-card {
        min-height: 150px;
        border-radius: 22px;
        padding: 20px;
        border: 1px solid #303030;
        background: #202020;
        position: relative;
    }

    .metric-green {
        background: linear-gradient(145deg, #35e59a, #27c982);
        color: #07130d;
        border: none;
    }

    .metric-gray {
        background: linear-gradient(145deg, #868686, #6f6f6f);
    }

    .metric-label {
        font-size: 12px;
        opacity: .78;
        margin-bottom: 16px;
    }

    .metric-name {
        font-size: 15px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 35px;
        font-weight: 500;
        letter-spacing: -1.4px;
    }

    .metric-arrow {
        position: absolute;
        top: 18px;
        right: 18px;
        height: 44px;
        width: 44px;
        border-radius: 50%;
        background: #f5f5f5;
        color: #111;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 22px;
    }

    .mini-label {
        color: #8c8c8c;
        font-size: 12px;
    }

    .mini-value {
        color: #f7f7f7;
        font-size: 25px;
        margin-top: 4px;
    }

    .status-ok {
        color: #35e59a;
        font-weight: 700;
    }

    .status-bad {
        color: #ff6b6b;
        font-weight: 700;
    }

    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: #121212;
        border: 1px solid #353535;
        color: #fff;
        border-radius: 15px;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: #121212;
        border-color: #353535;
        border-radius: 14px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #303030;
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background: #141414;
        border: 1px solid #303030;
        border-radius: 15px;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 15px;
        border: 1px solid #35e59a;
        background: #35e59a;
        color: #06120c;
        font-weight: 800;
    }

    .stButton > button:hover {
        background: #57efa9;
        color: #06120c;
        border-color: #57efa9;
    }

    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        min-height: 46px;
        border-radius: 15px;
        background: #202020;
        color: #fff;
        border: 1px solid #454545;
    }

    div[role="radiogroup"] {
        background: #202020;
        padding: 5px;
        border-radius: 28px;
        display: flex;
        gap: 4px;
    }

    div[role="radiogroup"] label {
        padding: 7px 15px;
        border-radius: 22px;
    }

    @media (max-width: 900px) {
        .hero-title {font-size: 35px;}
        .hero-subtitle {font-size: 29px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONNECTIONS
# ============================================================

@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def query_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with get_engine().connect() as connection:
        return pd.read_sql(text(sql), connection, params=params)


@st.cache_resource
def load_vector_store() -> FAISS:
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_DIR}. "
            "Run build_vector_store.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.1)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=300)
def get_dashboard_metrics() -> dict[str, Any]:
    df = query_df(
        """
        select
            total_transactions,
            actual_fraud_transactions,
            fraud_rate_percent,
            fraud_amount,
            avg_transaction_amount,
            max_transaction_amount
        from fraud_dashboard_metrics
        limit 1
        """
    )

    if df.empty:
        raise ValueError("fraud_dashboard_metrics returned no rows.")

    return df.iloc[0].to_dict()


@st.cache_data(ttl=300)
def get_type_stats() -> pd.DataFrame:
    return query_df(
        """
        select
            transaction_type,
            count(*) as transaction_count,
            sum(is_fraud) as fraud_count,
            avg(amount) as average_amount,
            sum(case when is_fraud = 1 then amount else 0 end) as fraud_amount
        from fact_transactions
        group by transaction_type
        order by fraud_count desc, transaction_count desc
        """
    )


@st.cache_data(ttl=300)
def get_step_trend() -> pd.DataFrame:
    return query_df(
        """
        select
            step,
            count(*) as total_transactions,
            sum(is_fraud) as fraud_transactions,
            sum(is_flagged_fraud) as flagged_transactions
        from fact_transactions
        group by step
        order by step
        """
    )


@st.cache_data(ttl=300)
def get_prediction_summary() -> pd.DataFrame:
    return query_df(
        """
        select
            predicted_fraud,
            actual_fraud,
            transaction_count,
            avg_fraud_probability,
            min_fraud_probability,
            max_fraud_probability
        from fraud_prediction_summary
        order by actual_fraud, predicted_fraud
        """
    )


@st.cache_data(ttl=120)
def get_transactions(
    transaction_type: str,
    minimum_amount: float,
    only_fraud: bool,
    limit: int,
) -> pd.DataFrame:
    filters = ["amount >= :minimum_amount"]
    params: dict[str, Any] = {
        "minimum_amount": minimum_amount,
        "limit": limit,
    }

    if transaction_type != "All":
        filters.append("transaction_type = :transaction_type")
        params["transaction_type"] = transaction_type

    if only_fraud:
        filters.append("is_fraud = 1")

    where_clause = " and ".join(filters)

    return query_df(
        f"""
        select
            transaction_key,
            step,
            transaction_type,
            amount,
            old_balance_origin,
            new_balance_origin,
            old_balance_destination,
            new_balance_destination,
            transaction_to_balance_ratio,
            high_amount_flag,
            emptied_origin_account_flag,
            suspicious_transfer_flag,
            is_fraud,
            is_flagged_fraud
        from fraud_features
        where {where_clause}
        order by amount desc
        limit :limit
        """,
        params=params,
    )


@st.cache_data(ttl=120)
def get_transaction_by_id(transaction_key: int) -> pd.DataFrame:
    return query_df(
        """
        select *
        from fraud_features
        where transaction_key = :transaction_key
        limit 1
        """,
        params={"transaction_key": transaction_key},
    )


# ============================================================
# RAG
# ============================================================

def build_context(documents) -> tuple[str, list[str]]:
    sections: list[str] = []
    sources: list[str] = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown")
        sources.append(source)
        sections.append(
            f"""Retrieved Fraud Case {index}
Source: {source}

{document.page_content}"""
        )

    return "\n\n".join(sections), sources


def generate_report(question: str, context: str) -> str:
    prompt = f"""
You are an enterprise financial fraud investigation assistant.

Use only the retrieved fraud-case context below.

Rules:
- Do not invent transaction details.
- Separate observed evidence from interpretation.
- Do not claim historical similarity proves fraud.
- State when evidence is unavailable.
- Write for a professional fraud operations team.

Return exactly:

### Investigation Summary
<brief conclusion>

### Main Risk Indicators
- <indicator>
- <indicator>
- <indicator>

### Similar Historical Patterns
- <pattern>
- <pattern>

### Recommended Investigator Actions
- <action>
- <action>
- <action>

Investigator Question:
{question}

Retrieved Context:
{context}
"""

    response = load_llm().invoke(prompt)
    return str(response.content)


# ============================================================
# CHARTS
# ============================================================

def trend_chart(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    if not df.empty:
        figure.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["total_transactions"],
                mode="lines",
                name="Transactions",
                line={"width": 2, "color": "#35e59a"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["fraud_transactions"],
                mode="lines",
                name="Fraud",
                line={"width": 2, "color": "#eeeeee"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["flagged_transactions"],
                mode="lines",
                name="Flagged",
                line={"width": 2, "color": "#5c86b3"},
            )
        )

    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font={"color": "#bdbdbd"},
        xaxis={"title": "Step", "showgrid": False, "zeroline": False},
        yaxis={"title": "Records", "gridcolor": "#343434", "zeroline": False},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    return figure


def fraud_type_chart(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=df["transaction_type"],
            y=df["fraud_count"],
            text=df["fraud_count"],
            textposition="outside",
            marker={"color": "#35e59a"},
        )
    )

    figure.update_layout(
        height=350,
        margin={"l": 10, "r": 10, "t": 25, "b": 15},
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font={"color": "#bdbdbd"},
        xaxis={"title": "Transaction type", "showgrid": False},
        yaxis={"title": "Fraud count", "gridcolor": "#343434"},
    )
    return figure


def confusion_chart(summary: pd.DataFrame) -> go.Figure:
    matrix = {
        "True negative": 0,
        "False positive": 0,
        "False negative": 0,
        "True positive": 0,
    }

    for _, row in summary.iterrows():
        predicted = int(row["predicted_fraud"])
        actual = int(row["actual_fraud"])
        count = int(row["transaction_count"])

        if actual == 0 and predicted == 0:
            matrix["True negative"] = count
        elif actual == 0 and predicted == 1:
            matrix["False positive"] = count
        elif actual == 1 and predicted == 0:
            matrix["False negative"] = count
        elif actual == 1 and predicted == 1:
            matrix["True positive"] = count

    figure = go.Figure(
        go.Bar(
            x=list(matrix.keys()),
            y=list(matrix.values()),
            text=list(matrix.values()),
            textposition="outside",
            marker={"color": ["#35e59a", "#ffb347", "#ff6b6b", "#eeeeee"]},
        )
    )

    figure.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="#181818",
        plot_bgcolor="#181818",
        font={"color": "#bdbdbd"},
        xaxis={"showgrid": False},
        yaxis={"title": "Predictions", "gridcolor": "#343434"},
    )
    return figure


# ============================================================
# UI HELPERS
# ============================================================

def refresh_app() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


def render_header() -> str:
    left, center, right = st.columns([1.2, 4.2, 1.1], vertical_alignment="center")

    with left:
        st.markdown('<div class="brand">FraudShield</div>', unsafe_allow_html=True)

    with center:
        page = st.radio(
            "Navigation",
            ["Analytics", "Transactions", "AI Investigator", "Model Monitoring", "System"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with right:
        if st.button("Refresh data", key="global_refresh"):
            refresh_app()

    return page


def render_kpi_card(
    label: str,
    name: str,
    value: str,
    card_class: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            <div class="metric-arrow">↗</div>
            <div class="metric-label">{label}</div>
            <div class="metric-name">{name}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================

page = render_header()

try:
    metrics = get_dashboard_metrics()
except Exception as error:
    st.error(
        "The dashboard could not connect to PostgreSQL or read the dbt views. "
        "Confirm Docker is running and port 55432 is available."
    )
    st.exception(error)
    st.stop()


# ---------------- ANALYTICS ----------------
if page == "Analytics":
    st.markdown(
        '<div class="hero-title">Risk & Fraud</div>'
        '<div class="hero-subtitle">Analytics</div>',
        unsafe_allow_html=True,
    )

    total = int(metrics["total_transactions"])
    fraud = int(metrics["actual_fraud_transactions"])
    fraud_rate = float(metrics["fraud_rate_percent"])
    legitimate_rate = max(0.0, 100.0 - fraud_rate)

    title_col, c1, c2, c3 = st.columns([1.25, 1, 1, 1], vertical_alignment="center")

    with title_col:
        st.markdown(
            "<div class='mini-label'>Live PostgreSQL warehouse</div>"
            "<div class='mini-value'>6.36M+ transaction intelligence</div>",
            unsafe_allow_html=True,
        )

    with c1:
        render_kpi_card(
            "Fraud screening rate",
            "Confirmed fraud",
            f"{fraud_rate:.4f}%",
            "metric-green",
        )

    with c2:
        render_kpi_card(
            "Legitimate transaction rate",
            "Authenticated",
            f"{legitimate_rate:.2f}%",
            "metric-gray",
        )

    with c3:
        render_kpi_card(
            "Transactions reviewed",
            "Total volume",
            f"{total:,}",
        )

    st.write("")

    stat_cols = st.columns(6)
    stat_values = [
        ("Total transactions", f"{total:,}"),
        ("Confirmed fraud", f"{fraud:,}"),
        ("Legitimate", f"{total - fraud:,}"),
        ("Fraud rate", f"{fraud_rate:.4f}%"),
        ("Average amount", f"${float(metrics['avg_transaction_amount']):,.0f}"),
        ("Maximum amount", f"${float(metrics['max_transaction_amount']):,.0f}"),
    ]

    for column, (label, value) in zip(stat_cols, stat_values):
        with column:
            st.markdown(
                f"<div class='mini-label'>{label}</div>"
                f"<div class='mini-value'>{value}</div>",
                unsafe_allow_html=True,
            )

    try:
        trend = get_step_trend()
        type_stats = get_type_stats()

        left_chart, right_chart = st.columns([2.1, 1])

        with left_chart:
            st.markdown('<div class="section-title">Transaction risk trend</div>', unsafe_allow_html=True)
            st.plotly_chart(
                trend_chart(trend),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with right_chart:
            st.markdown('<div class="section-title">Fraud by type</div>', unsafe_allow_html=True)
            st.plotly_chart(
                fraud_type_chart(type_stats),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown('<div class="section-title">Transaction-type summary</div>', unsafe_allow_html=True)
        st.dataframe(type_stats, use_container_width=True, hide_index=True)

    except SQLAlchemyError as error:
        st.error(f"Unable to load analytics data: {error}")


# ---------------- TRANSACTIONS ----------------
elif page == "Transactions":
    st.markdown(
        '<div class="hero-title">Transaction</div>'
        '<div class="hero-subtitle">Explorer</div>',
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(4)

    with filter_cols[0]:
        transaction_type = st.selectbox(
            "Transaction type",
            ["All", "TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
        )

    with filter_cols[1]:
        minimum_amount = st.number_input(
            "Minimum amount",
            min_value=0.0,
            value=100000.0,
            step=10000.0,
        )

    with filter_cols[2]:
        only_fraud = st.checkbox("Only confirmed fraud", value=False)

    with filter_cols[3]:
        limit = st.selectbox("Rows", [25, 50, 100, 250, 500], index=1)

    action_cols = st.columns([1, 1, 3])

    with action_cols[0]:
        search_clicked = st.button("Run search", key="transaction_search")

    with action_cols[1]:
        if st.button("Reset filters", key="reset_filters"):
            st.session_state.clear()
            st.rerun()

    if search_clicked or "transaction_results" not in st.session_state:
        try:
            st.session_state.transaction_results = get_transactions(
                transaction_type=transaction_type,
                minimum_amount=float(minimum_amount),
                only_fraud=only_fraud,
                limit=int(limit),
            )
        except Exception as error:
            st.error(f"Transaction search failed: {error}")
            st.session_state.transaction_results = pd.DataFrame()

    results = st.session_state.get("transaction_results", pd.DataFrame())

    if results.empty:
        st.info("No transactions matched the selected filters.")
    else:
        st.success(f"Found {len(results):,} matching transactions.")
        st.dataframe(results, use_container_width=True, hide_index=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_data,
            file_name="fraudshield_transactions.csv",
            mime="text/csv",
        )

    st.markdown('<div class="section-title">Lookup one transaction</div>', unsafe_allow_html=True)

    lookup_cols = st.columns([1, 1, 3])

    with lookup_cols[0]:
        transaction_key = st.number_input(
            "Transaction key",
            min_value=1,
            step=1,
            value=1,
        )

    with lookup_cols[1]:
        lookup_clicked = st.button("Lookup transaction", key="lookup_transaction")

    if lookup_clicked:
        try:
            detail = get_transaction_by_id(int(transaction_key))

            if detail.empty:
                st.warning("No transaction was found with that key.")
            else:
                st.dataframe(detail, use_container_width=True, hide_index=True)

                row = detail.iloc[0]
                risk_flags = int(row.get("high_amount_flag", 0)) + int(
                    row.get("emptied_origin_account_flag", 0)
                ) + int(row.get("suspicious_transfer_flag", 0))

                if int(row.get("is_fraud", 0)) == 1:
                    st.error(f"Confirmed fraud. Active risk flags: {risk_flags}")
                elif risk_flags > 0:
                    st.warning(f"Not labeled fraud, but {risk_flags} risk flags are active.")
                else:
                    st.success("No engineered risk flags are active.")
        except Exception as error:
            st.error(f"Transaction lookup failed: {error}")


# ---------------- AI INVESTIGATOR ----------------
elif page == "AI Investigator":
    st.markdown(
        '<div class="hero-title">AI Fraud</div>'
        '<div class="hero-subtitle">Investigator</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1])

    with left:
        question = st.text_area(
            "Investigation question",
            placeholder=(
                "Why are high-value transfers that empty the origin "
                "account considered suspicious?"
            ),
            height=170,
        )

        case_count = st.slider(
            "Historical cases to retrieve",
            min_value=1,
            max_value=10,
            value=5,
        )

        investigate_clicked = st.button(
            "Generate investigation report",
            key="investigate",
        )

        clear_clicked = st.button(
            "Clear investigation",
            key="clear_investigation",
        )

        if clear_clicked:
            st.session_state.pop("investigation_report", None)
            st.session_state.pop("retrieved_documents", None)
            st.rerun()

    with right:
        col1, col2 = st.columns(2)
        col1.metric("Knowledge base", "1,000 cases")
        col2.metric("Vector store", "FAISS")

        col3, col4 = st.columns(2)
        col3.metric("Embedding", "MiniLM-L6")
        col4.metric("Local LLM", "Llama 3.2")

        if st.button("Check Ollama", key="check_ollama"):
            try:
                response = load_llm().invoke("Reply with exactly: Ollama is ready.")
                st.success(str(response.content))
            except Exception as error:
                st.error(f"Ollama check failed: {error}")

    if investigate_clicked:
        if not question.strip():
            st.warning("Enter an investigation question.")
        else:
            try:
                with st.spinner("Retrieving similar fraud cases..."):
                    documents = load_vector_store().similarity_search(
                        question,
                        k=int(case_count),
                    )

                context, _ = build_context(documents)

                with st.spinner("Generating grounded investigator report..."):
                    report = generate_report(question, context)

                st.session_state.investigation_report = report
                st.session_state.retrieved_documents = documents

            except Exception as error:
                st.error(f"Investigation failed: {error}")

    report = st.session_state.get("investigation_report")
    documents = st.session_state.get("retrieved_documents", [])

    if report:
        report_col, evidence_col = st.columns([1.5, 1])

        with report_col:
            st.markdown('<div class="section-title">Investigation report</div>', unsafe_allow_html=True)
            st.markdown(report)

            st.download_button(
                "Download investigation report",
                data=report.encode("utf-8"),
                file_name="fraud_investigation_report.md",
                mime="text/markdown",
            )

        with evidence_col:
            st.markdown('<div class="section-title">Retrieved evidence</div>', unsafe_allow_html=True)

            for index, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "Unknown")

                with st.expander(f"Case {index}: {source}"):
                    case_path = CASE_SUMMARY_DIR / source

                    if case_path.exists():
                        st.text(case_path.read_text(encoding="utf-8"))
                    else:
                        st.text(document.page_content)


# ---------------- MODEL MONITORING ----------------
elif page == "Model Monitoring":
    st.markdown(
        '<div class="hero-title">Model</div>'
        '<div class="hero-subtitle">Monitoring</div>',
        unsafe_allow_html=True,
    )

    if st.button("Refresh model metrics", key="refresh_model_metrics"):
        get_prediction_summary.clear()
        st.rerun()

    try:
        summary = get_prediction_summary()

        totals = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

        for _, row in summary.iterrows():
            predicted = int(row["predicted_fraud"])
            actual = int(row["actual_fraud"])
            count = int(row["transaction_count"])

            if actual == 0 and predicted == 0:
                totals["tn"] = count
            elif actual == 0 and predicted == 1:
                totals["fp"] = count
            elif actual == 1 and predicted == 0:
                totals["fn"] = count
            else:
                totals["tp"] = count

        precision = (
            totals["tp"] / (totals["tp"] + totals["fp"])
            if totals["tp"] + totals["fp"]
            else 0
        )
        recall = (
            totals["tp"] / (totals["tp"] + totals["fn"])
            if totals["tp"] + totals["fn"]
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Fraud precision", f"{precision:.4f}")
        metric_cols[1].metric("Fraud recall", f"{recall:.4f}")
        metric_cols[2].metric("Fraud F1", f"{f1:.4f}")
        metric_cols[3].metric("False positives", f"{totals['fp']:,}")
        metric_cols[4].metric("False negatives", f"{totals['fn']:,}")

        st.plotly_chart(
            confusion_chart(summary),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.dataframe(summary, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Model monitoring failed: {error}")


# ---------------- SYSTEM ----------------
elif page == "System":
    st.markdown(
        '<div class="hero-title">Platform</div>'
        '<div class="hero-subtitle">System Health</div>',
        unsafe_allow_html=True,
    )

    if st.button("Run health checks", key="run_health_checks"):
        results: list[tuple[str, bool, str]] = []

        try:
            db_test = query_df("select current_database() as database_name")
            results.append(
                (
                    "PostgreSQL",
                    True,
                    f"Connected to {db_test.iloc[0]['database_name']}",
                )
            )
        except Exception as error:
            results.append(("PostgreSQL", False, str(error)))

        try:
            store = load_vector_store()
            document_count = len(store.index_to_docstore_id)
            results.append(
                (
                    "FAISS vector store",
                    True,
                    f"{document_count:,} indexed documents",
                )
            )
        except Exception as error:
            results.append(("FAISS vector store", False, str(error)))

        try:
            llm_response = load_llm().invoke("Reply with READY only.")
            results.append(
                (
                    "Ollama / Llama 3.2",
                    True,
                    str(llm_response.content),
                )
            )
        except Exception as error:
            results.append(("Ollama / Llama 3.2", False, str(error)))

        results.append(
            (
                "Case summary folder",
                CASE_SUMMARY_DIR.exists(),
                (
                    f"{len(list(CASE_SUMMARY_DIR.glob('*.txt'))):,} files"
                    if CASE_SUMMARY_DIR.exists()
                    else "Folder not found"
                ),
            )
        )

        st.session_state.health_results = results

    for service, is_ok, message in st.session_state.get("health_results", []):
        if is_ok:
            st.success(f"{service}: {message}")
        else:
            st.error(f"{service}: {message}")

    st.markdown('<div class="section-title">Maintenance actions</div>', unsafe_allow_html=True)

    maintenance_cols = st.columns(3)

    with maintenance_cols[0]:
        if st.button("Clear Streamlit cache", key="clear_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Application cache cleared.")

    with maintenance_cols[1]:
        if st.button("Reload vector store", key="reload_vector"):
            load_vector_store.clear()
            st.success("Vector-store cache cleared. It will reload on the next request.")

    with maintenance_cols[2]:
        if st.button("Reload LLM connection", key="reload_llm"):
            load_llm.clear()
            st.success("LLM cache cleared. It will reconnect on the next request.")

    st.info(
        "For safety, this interface does not automatically retrain the model, "
        "rebuild dbt tables, or overwrite the vector index. Run those controlled "
        "pipeline commands from the terminal."
    )
