import streamlit as st
import yfinance as yf
import joblib
import plotly.graph_objects as go
import numpy as np

from utils import add_indicators

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

period = st.sidebar.selectbox(
    "Select Period",
    ["6mo", "1y", "2y", "5y"]
)

# ------------------ TITLE ------------------
st.title("📊 AI Stock Prediction Dashboard")

# ------------------ LOAD DATA ------------------
data = yf.download(ticker, period=period)

if data.empty:
    st.error("No data found!")
    st.stop()

data = add_indicators(data)

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

price = round(data['Close'].iloc[-1], 2)

col1.metric("💰 Current Price", price)

# ------------------ LOAD MODEL ------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

features = [
'SMA','Momentum','RSI','EMA',
'MACD','BB_upper','BB_lower',
'ATR','OBV'
]

X = data[features].tail(1)
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0][1]

# ------------------ SIGNAL ------------------
if prediction == 1:
    col2.metric("📈 Signal", "BUY")
else:
    col2.metric("📉 Signal", "SELL")

# ------------------ CONFIDENCE ------------------
col3.metric("🎯 Confidence", f"{prob*100:.2f}%")

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Indicators", "📉 Data"])

# ------------------ CANDLESTICK ------------------
with tab1:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ INDICATORS ------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI")
        st.line_chart(data['RSI'])

    with col2:
        st.subheader("MACD")
        st.line_chart(data['MACD'])

# ------------------ DATA TABLE ------------------
with tab3:
    st.dataframe(data.tail(50))

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Built by Ayush")