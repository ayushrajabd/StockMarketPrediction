import streamlit as st
from pathlib import Path

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Stock Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ DEPENDENCY CHECK ------------------
missing_packages = []

try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None

try:
    import joblib
except ModuleNotFoundError:
    missing_packages.append("joblib")

try:
    import pandas as pd
except ModuleNotFoundError:
    missing_packages.append("pandas")

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    missing_packages.append("plotly")

try:
    from sentiment import get_sentiment
except (ImportError, ModuleNotFoundError):
    def get_sentiment(stock_name):
        return 0

try:
    from utils import add_indicators
except ModuleNotFoundError as exc:
    missing_packages.append(exc.name)
except ImportError as exc:
    st.error("Could not import the project helper file utils.py.")
    st.caption(str(exc))
    st.stop()

APP_DIR = Path(__file__).resolve().parent

if missing_packages:
    packages = ", ".join(sorted(set(missing_packages)))
    st.error(f"Missing Python package(s): {packages}")
    st.write("Install the required packages, then run the app again:")
    st.code("pip install -r requirements.txt", language="powershell")
    st.stop()

# ------------------ SIDEBAR ------------------
st.sidebar.title("Settings")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").strip().upper()

period = st.sidebar.selectbox(
    "Select Period",
    ["6mo", "1y", "2y", "5y"],
)

if not ticker:
    st.error("Please enter a stock ticker.")
    st.stop()

# ------------------ SENTIMENT ------------------
sentiment_score = get_sentiment(ticker)
st.sidebar.metric("Market Sentiment", round(sentiment_score, 2))

# ------------------ TITLE ------------------
st.title("AI Stock Prediction Dashboard")

# ------------------ DATA ------------------
if yf is None:
    stock_data_path = APP_DIR / "stock_data.xlsx"
    if not stock_data_path.exists():
        st.error("yfinance is not installed and stock_data.xlsx is missing.")
        st.code("pip install yfinance", language="powershell")
        st.stop()

    st.info("yfinance is not installed, so the app is using saved stock_data.xlsx data.")
    try:
        data = pd.read_excel(stock_data_path)
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")
    except Exception as exc:
        st.error("Could not read stock_data.xlsx.")
        st.caption(str(exc))
        st.stop()
else:
    try:
        data = yf.download(ticker, period=period)
    except Exception as exc:
        st.error("Could not download stock data. Check your internet connection and ticker symbol.")
        st.caption(str(exc))
        st.stop()

if data.empty:
    st.error("No data found. Please check the ticker symbol and try again.")
    st.stop()

try:
    data = add_indicators(data)
except Exception as exc:
    st.error("Could not calculate stock indicators.")
    st.caption(str(exc))
    st.stop()

data["Sentiment"] = sentiment_score

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

price = round(data["Close"].iloc[-1], 2)

col1.metric("Current Price", price)
col2.metric("Sentiment", round(sentiment_score, 2))

# ------------------ LOAD MODEL ------------------
model_path = APP_DIR / "model.pkl"
scaler_path = APP_DIR / "scaler.pkl"

if not model_path.exists() or not scaler_path.exists():
    st.error("Model files are missing. Keep model.pkl and scaler.pkl in the same folder as app.py.")
    st.stop()

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as exc:
    st.error("Could not load the saved model files.")
    st.caption(str(exc))
    st.stop()

features = [
    "SMA",
    "Momentum",
    "RSI",
    "EMA",
    "MACD",
    "BB_upper",
    "BB_lower",
    "ATR",
    "OBV",
    "Sentiment",
]

missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    st.error(f"Missing feature columns: {', '.join(missing_features)}")
    st.stop()

X = data[features].tail(1)

try:
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
except Exception as exc:
    st.error("Could not make a prediction with the saved model.")
    st.caption(str(exc))
    st.stop()

# ------------------ SIGNAL + CONFIDENCE ------------------
col2, col3 = st.columns(2)

if prediction == 1:
    col2.metric("Signal", "BUY")
else:
    col2.metric("Signal", "SELL")

col3.metric("Confidence", f"{prob * 100:.2f}%")

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["Chart", "Indicators", "Data"])

# ------------------ CHART ------------------
with tab1:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
        )
    )

    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------ INDICATORS ------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI")
        st.line_chart(data["RSI"])

    with col2:
        st.subheader("MACD")
        st.line_chart(data["MACD"])

# ------------------ DATA ------------------
with tab3:
    st.dataframe(data.tail(50))

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Built by Ayush and Vikas")
