import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from utils import add_indicators

from sentiment import get_sentiment


data = yf.download("AAPL", start="2015-01-01")

if data.empty:
    print("No data found. Check stock symbol or internet.")
    exit()


data = add_indicators(data)
stock_name = "AAPL"
sentiment_value = get_sentiment(stock_name)
data['Sentiment'] = sentiment_value

from sentiment import get_sentiment

data = add_indicators(data)

data['Sentiment'] = get_sentiment("AAPL")   # ✅ MUST be here
data = data.dropna()

features = [
'SMA','Momentum','RSI','EMA',
'MACD','BB_upper','BB_lower',
'ATR','OBV','Sentiment'
]

X = data[features]
data = data.dropna()
y = data['Target']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier(n_estimators=300)
model.fit(X_scaled, y)


joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved")

data.to_excel("stock_data.xlsx")
print("File saved!")