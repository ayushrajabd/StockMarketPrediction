# sentiment.py
from textblob import TextBlob
import random

def get_sentiment(stock_name):
    try:
        # Dynamic sample data (temporary fix)
        tweets_pool = [
            f"{stock_name} is performing very well",
            f"{stock_name} stock is crashing badly",
            f"{stock_name} has strong growth potential",
            f"{stock_name} is risky investment",
            f"{stock_name} is stable and reliable",
            f"{stock_name} facing losses",
            f"{stock_name} is booming in market",
            f"{stock_name} may fall soon"
        ]

        # Pick random tweets
        tweets = random.sample(tweets_pool, 4)

        polarity_scores = [
            TextBlob(tweet).sentiment.polarity for tweet in tweets
        ]

        return sum(polarity_scores) / len(polarity_scores)

    except Exception as e:
        print("Sentiment Error:", e)
        return 0