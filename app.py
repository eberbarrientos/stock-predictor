import streamlit as st
import requests

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
        # Short-Term (Stock model)
        short_resp = requests.get(f"{api_url}/predict_stock?ticker={ticker}")
        short_data = short_resp.json()

        st.subheader("⚡ Short-Term Model (Quick Gains Focus)")
        st.write(f"Confidence: **{short_data.get('stock_confidence', 'N/A')}**")
        st.write(f"Signal: **{short_data.get('signal', 'N/A')}**")

        # Long-Term (ETF model), only if horizon is selected
        if horizon != "None":
            long_resp = requests.get(f"{api_url}/predict_etf?ticker={ticker}&horizon={horizon}")
            long_data = long_resp.json()

            st.subheader("📈 Long-Term Model (Wealth Builder Focus)")
            st.write(f"Confidence: **{long_data.get('etf_confidence', 'N/A')}**")
            st.write(f"Signal: **{long_data.get('signal', 'N/A')}**")

            st.markdown("""
            🧠 *Tip:* Look for **agreement between both models** for highest confidence in your move.
            """)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
