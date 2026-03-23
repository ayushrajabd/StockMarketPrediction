import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from utils import add_indicators


data = yf.download("AAPL", start="2015-01-01")

if data.empty:
    print("No data found. Check stock symbol or internet.")
    exit()


data = add_indicators(data)

features = [
'SMA','Momentum','RSI','EMA',
'MACD','BB_upper','BB_lower',
'ATR','OBV'
]

X = data[features]
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