import streamlit as st
import requests
import datetime

st.markdown("### 🛠️ Optional: Train Model on Custom Date Range")

custom_train = st.checkbox("Enable Custom Training Period")
train_start = train_end = None
params_stock = {}
params_etf = {}

if custom_train:
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input("Training Start Date", datetime.date(2015, 1, 1))
    with col2:
        train_end = st.date_input("Training End Date", datetime.date(2023, 12, 31))

    # Add training dates to params
    params_stock["train_start"] = train_start.strftime("%Y-%m-%d")
    params_stock["train_end"] = train_end.strftime("%Y-%m-%d")
    params_etf["train_start"] = train_start.strftime("%Y-%m-%d")
    params_etf["train_end"] = train_end.strftime("%Y-%m-%d")

api_url = "https://stock-predictor-production.up.railway.app"

st.title("📊 Smart Investment Signal Dashboard")

st.markdown("""
Welcome to your dual-strategy investment assistant.  
Use the **Short-Term Model** to catch quick opportunities, and the **Long-Term Model** to align with consistent growth strategies.
""")

ticker = st.text_input("🔎 Enter Stock or ETF Ticker", "AAPL")
horizon_options = ["None", 30, 60, 90, 120, 180]
horizon = st.selectbox("📆 Optional: Long-Term Horizon (days)", horizon_options)

if st.button("📈 Get Smart Investment Signals"):
    try:
        # Add ticker to params
        params_stock["ticker"] = ticker

        # --- Short-Term Model ---
        short_resp = requests.get(f"{api_url}/predict_stock", params=params_stock)
        short_data = short_resp.json()

        st.subheader("⚡ Short-Term Model (Quick Gains Focus)")
        st.write(f"Confidence: **{short_data.get('stock_confidence', 'N/A')}**")
        st.write(f"Signal: **{short_data.get('signal', 'N/A')}**")

        # --- Long-Term Model ---
        if horizon != "None":
            params_etf["ticker"] = ticker
            params_etf["horizon"] = horizon
            long_resp = requests.get(f"{api_url}/predict_etf", params=params_etf)
            long_data = long_resp.json()

            st.subheader("📈 Long-Term Model (Wealth Builder Focus)")
            st.write(f"Confidence: **{long_data.get('etf_confidence', 'N/A')}**")
            st.write(f"Signal: **{long_data.get('signal', 'N/A')}**")

            st.markdown("""
            🧠 *Tip:* Look for **agreement between both models** for highest confidence in your move.
            """)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
