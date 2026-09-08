import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 10
DECISION_THRESHOLD = 0.5

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      /* Colours are expressed as translucent accents over the page background
         so the panel reads correctly in both the light and the dark theme. */
      .block-container { padding-top: 3rem; padding-bottom: 2rem; }

      .app-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
      .app-sub   { font-size: 0.9rem; margin-top: .15rem; opacity: .65; }

      .pill {
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .3rem .7rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600; white-space: nowrap;
      }
      .pill-ok   { background: rgba(22, 163, 74, .16);  color: #22c55e; }
      .pill-warn { background: rgba(220, 38, 38, .16);  color: #ef4444; }
      .dot { width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }

      .section-label {
        font-size: .72rem; font-weight: 700; letter-spacing: .08em;
        text-transform: uppercase; opacity: .55;
        margin: .2rem 0 .5rem 0;
      }

      .verdict {
        border-radius: 14px; padding: 1.4rem 1.5rem; margin-bottom: 1rem;
        border: 1px solid; text-align: center;
      }
      .verdict-high { background: rgba(220, 38, 38, .10); border-color: rgba(220, 38, 38, .35); }
      .verdict-low  { background: rgba(22, 163, 74, .10); border-color: rgba(22, 163, 74, .35); }
      .verdict-tag  { font-size: .78rem; font-weight: 700; letter-spacing: .1em;
                      text-transform: uppercase; }
      .verdict-high .verdict-tag { color: #ef4444; }
      .verdict-low  .verdict-tag { color: #22c55e; }
      .verdict-num  { font-size: 2.9rem; font-weight: 800; line-height: 1.1; margin: .3rem 0 .1rem 0; }
      .verdict-high .verdict-num { color: #ef4444; }
      .verdict-low  .verdict-num { color: #22c55e; }
      .verdict-cap  { font-size: .8rem; opacity: .6; }

      .risk-track {
        position: relative; height: 12px; border-radius: 999px;
        background: rgba(128, 128, 128, .25); overflow: hidden; margin: .2rem 0 .35rem 0;
      }
      .risk-fill { height: 100%; border-radius: 999px; }
      .risk-scale {
        display: flex; justify-content: space-between;
        font-size: .7rem; opacity: .5;
      }

      .meta-row {
        display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .9rem;
      }
      .chip {
        background: rgba(128, 128, 128, .15); border-radius: 8px;
        padding: .3rem .6rem; font-size: .75rem; opacity: .85;
      }
      .chip b { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None


def friendly_error(exc):
    """Turn a requests exception into something a dashboard user can act on."""
    if isinstance(exc, requests.ConnectionError):
        return (
            f"No API is listening at {API_URL}. Start it with "
            "`uv run uvicorn src.api.main:app --port 8000`."
        )
    if isinstance(exc, requests.Timeout):
        return f"The API at {API_URL} did not respond within {TIMEOUT} seconds."
    return f"Request to {API_URL} failed: {exc}"


@st.cache_data(ttl=10, show_spinner=False)
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        return r.json()
    except requests.RequestException:
        return None


def fetch_logs(limit=10):
    try:
        r = requests.get(f"{API_URL}/logs", params={"limit": limit}, timeout=TIMEOUT)
        r.raise_for_status()
        return pd.DataFrame(r.json()), None
    except requests.RequestException as e:
        return None, friendly_error(e)


head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown('<p class="app-title">🏦 Customer Churn Risk</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-sub">Score a customer against the CatBoost churn model '
        "and keep an audit trail of every prediction.</p>",
        unsafe_allow_html=True,
    )
with head_right:
    health = api_health()
    if health and health.get("status") == "ok":
        st.markdown(
            '<div style="text-align:right"><span class="pill pill-ok">'
            '<span class="dot"></span>API healthy</span></div>',
            unsafe_allow_html=True,
        )
    else:
        label = "API unreachable" if health is None else "API degraded"
        st.markdown(
            f'<div style="text-align:right"><span class="pill pill-warn">'
            f'<span class="dot"></span>{label}</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

col_left, col_right = st.columns([1.35, 1], gap="large")

with col_left:
    with st.form("churn_prediction_form"):
        st.markdown('<p class="section-label">Identity</p>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        with i1:
            customer_id = st.number_input(
                "Customer ID", min_value=0, step=1, value=0,
                help="Stored with the prediction so the outcome can be matched later.",
            )
        with i2:
            surname = st.text_input("Surname", value="", placeholder="Yilmaz")

        st.markdown('<p class="section-label">Profile</p>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=None)
        with p2:
            gender = st.selectbox("Gender", ["Female", "Male"], index=None)
        with p3:
            age = st.number_input("Age", 18, 100, value=None)

        st.markdown('<p class="section-label">Account</p>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            credit_score = st.number_input("Credit Score", 300, 850, value=None)
            balance = st.number_input("Balance", min_value=0.0, step=1000.0, value=None)
            tenure = st.number_input("Tenure (years)", 0, 10, value=None)
        with a2:
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0, value=None)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=None)
            card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "DIAMOND", "PLATINUM"], index=None)

        st.markdown('<p class="section-label">Engagement</p>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns([1, 1, 1.2])
        with e1:
            satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5], index=None)
        with e2:
            point_earned = st.number_input("Points Earned", min_value=0, step=10, value=None)
        with e3:
            st.write("")
            has_cr_card = st.checkbox("Has credit card", value=False)
            is_active = st.checkbox("Active member", value=False)

        st.write("")
        submit_btn = st.form_submit_button(
            "Analyze risk", type="primary", use_container_width=True
        )

if submit_btn:
    required = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Credit Score": credit_score,
        "Balance": balance,
        "Tenure": tenure,
        "Estimated Salary": estimated_salary,
        "Number of Products": num_products,
        "Card Type": card_type,
        "Satisfaction Score": satisfaction_score,
        "Points Earned": point_earned,
    }
    missing = [name for name, value in required.items() if value is None]

    if missing:
        st.session_state.result = {"error": "Missing fields: " + ", ".join(missing)}
    else:
        payload = {
            "CustomerId": int(customer_id),
            "Surname": surname.strip() or "Unknown",
            "CreditScore": int(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_products),
            "HasCrCard": 1 if has_cr_card else 0,
            "IsActiveMember": 1 if is_active else 0,
            "EstimatedSalary": float(estimated_salary),
            "CardType": card_type,
            "SatisfactionScore": int(satisfaction_score),
            "PointEarned": int(point_earned),
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
            if response.status_code == 200:
                body = response.json()
                st.session_state.result = {
                    "prediction": body["prediction"],
                    "probability": body["churn_probability"],
                    "log_id": body["log_id"],
                    "geography": geography,
                    "age": int(age),
                    "products": int(num_products),
                    "active": bool(is_active),
                }
            elif response.status_code == 422:
                details = response.json().get("detail", [])
                fields = ", ".join(".".join(str(p) for p in d.get("loc", [])[1:]) for d in details)
                st.session_state.result = {"error": f"The API rejected these fields: {fields}"}
            else:
                st.session_state.result = {
                    "error": f"Server returned {response.status_code}: {response.text[:200]}"
                }
        except requests.RequestException as e:
            st.session_state.result = {"error": friendly_error(e)}

with col_right:
    tab_result, tab_history = st.tabs(["Analysis result", "Prediction history"])

    with tab_result:
        result = st.session_state.result

        if result is None:
            st.info("Fill in the customer details and select **Analyze risk**.")
        elif "error" in result:
            st.error(result["error"])
        else:
            prob = result["probability"]
            high = result["prediction"] == "CHURN"
            css = "verdict-high" if high else "verdict-low"
            tag = "High risk · likely to churn" if high else "Low risk · likely to stay"
            colour = "#ef4444" if high else "#22c55e"

            st.markdown(
                f"""
                <div class="verdict {css}">
                  <div class="verdict-tag">{tag}</div>
                  <div class="verdict-num">{prob * 100:.1f}%</div>
                  <div class="verdict-cap">estimated churn probability</div>
                </div>
                <div class="risk-track">
                  <div class="risk-fill" style="width:{prob * 100:.1f}%;background:{colour}"></div>
                </div>
                <div class="risk-scale"><span>0%</span>
                  <span>decision threshold {DECISION_THRESHOLD:.0%}</span><span>100%</span></div>
                <div class="meta-row">
                  <span class="chip">Log <b>#{result['log_id']}</b></span>
                  <span class="chip">Geography <b>{result['geography']}</b></span>
                  <span class="chip">Age <b>{result['age']}</b></span>
                  <span class="chip">Products <b>{result['products']}</b></span>
                  <span class="chip">Active <b>{'yes' if result['active'] else 'no'}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Labelled **{result['prediction']}** because the probability is "
                f"{'above' if high else 'at or below'} the {DECISION_THRESHOLD:.0%} decision threshold."
            )

    with tab_history:
        st.button("Refresh", use_container_width=True)
        logs_df, error = fetch_logs(limit=10)

        if error is not None:
            st.warning(error)
        elif logs_df is None or logs_df.empty:
            st.info("No predictions recorded yet.")
        else:
            # Kept deliberately narrow: this table lives in the right-hand
            # column, and a wider set of columns clips the probability bar.
            view = logs_df[
                ["id", "created_at", "surname", "geography",
                 "prediction_label", "churn_probability"]
            ].copy()
            view["created_at"] = pd.to_datetime(view["created_at"]).dt.strftime("%d %b %H:%M")
            # ProgressColumn formats the raw value, so scale to percent here
            # rather than rendering 0.86 as "0.9%".
            view["churn_probability"] = view["churn_probability"] * 100

            st.dataframe(
                view,
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "id": st.column_config.NumberColumn("#", width="small"),
                    "created_at": st.column_config.TextColumn("Time", width="small"),
                    "surname": st.column_config.TextColumn("Surname", width="small"),
                    "geography": st.column_config.TextColumn("Country", width="small"),
                    "prediction_label": st.column_config.TextColumn("Verdict", width="small"),
                    "churn_probability": st.column_config.ProgressColumn(
                        "Risk", min_value=0.0, max_value=100.0, format="%.1f%%"
                    ),
                },
            )
            st.caption(f"Showing the {len(view)} most recent predictions.")
