# sentiment.py
from textblob import TextBlob

def get_sentiment(stock_name):
    try:
        # Dummy tweets (for testing)
        tweets = [
            f"{stock_name} is performing very well",
            f"{stock_name} stock is going down",
            f"I think {stock_name} has good future"
        ]

        polarity_scores = []

        for tweet in tweets:
            analysis = TextBlob(tweet)
            polarity_scores.append(analysis.sentiment.polarity)

        # Avoid division by zero
        if len(polarity_scores) == 0:
            return 0

        return sum(polarity_scores) / len(polarity_scores)

    except Exception as e:
        return 0