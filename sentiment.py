import random


def get_sentiment(stock_name):
    """Return a small demo sentiment score without requiring external APIs."""
    try:
        tweets_pool = [
            f"{stock_name} is performing very well",
            f"{stock_name} stock is crashing badly",
            f"{stock_name} has strong growth potential",
            f"{stock_name} is risky investment",
            f"{stock_name} is stable and reliable",
            f"{stock_name} facing losses",
            f"{stock_name} is booming in market",
            f"{stock_name} may fall soon",
        ]

        positive_words = {"well", "strong", "growth", "stable", "reliable", "booming"}
        negative_words = {"crashing", "badly", "risky", "losses", "fall"}

        tweets = random.sample(tweets_pool, 4)
        scores = []

        for tweet in tweets:
            words = set(tweet.lower().split())
            positive_count = len(words & positive_words)
            negative_count = len(words & negative_words)
            scores.append((positive_count - negative_count) / 3)

        return sum(scores) / len(scores)

    except Exception as e:
        print("Sentiment Error:", e)
        return 0
