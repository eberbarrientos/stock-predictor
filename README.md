# 📊 Smart Investment Signal Dashboard

A dual-strategy machine learning system that gives **short-term** and **long-term** stock/ETF trading signals using confidence-aware models and real-time market data.

## 🚀 Features

✅ **Short-Term Model (Stock-Based)**  
- Focused on quick price movements  
- Predicts buy/sell signals using technical indicators  
- Trained on historical stock data with a Random Forest classifier

✅ **Long-Term Model (ETF-Based)**  
- Focused on consistent growth  
- Trained on broad ETFs (like VOO) using horizon-based targets (30–180 days)  
- Predicts long-term confidence in upward trends  

✅ **Confidence Scoring**  
- Both models output probabilities  
- Combined logic allows users to interpret market agreement/disagreement  

✅ **Live Dashboard**  
- Built with Streamlit  
- Real-time predictions via deployed FastAPI backend  
- Clean UI with intuitive inputs for traders and investors

---

## 🧠 How It Works

### 📁 Model Pipeline
Raw Ticker ---> Feature Engineering ---> ML Model

- Uses technical indicators: RSI, SMA, Volatility, Returns
- Trained using scikit-learn + SMOTE for balanced classes
- Separate models for short- and long-term views

---

## 🖥️ Live Demo

## 🔗 Streamlit Dashboard  
👉 [Smart Investment Signal Dashboard](https://stock-predictor12473.streamlit.app)

### 🔗 API Server  
👉 [https://stock-predictor-production.up.railway.app](https://stock-predictor-production.up.railway.app)

---

## 💻 Local Installation

### 📦 Requirements
- Python 3.9+
- pip

### 🛠️ Setup
```bash
git clone https://github.com/eber-barrientos/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt

### 🔮 Run Streamlit App
```bash
streamlit run app.py

### Run API Server (FastAPI)
uvicorn api.api:app --reload
📈 API Endpoints

GET /predict_stock?ticker=AAPL
→ Returns stock model confidence + signal
GET /predict_etf?ticker=VOO&horizon=90
→ Returns ETF model confidence + signal for a specific horizon
🧠 Future Improvements

Backtesting & returns visualization
Portfolio optimization tools
LSTM/transformer-based sequence models
Interactive chart overlays in Streamlit

📬 Contact
Built by Eber Barrientos
🧠 Computer Science @ UTA
Linkedin: https://www.linkedin.com/in/eber-barrientos/
Email: eberbarr.com@gmail.com