import PhineasSentimentAnalyzer as psa

if __name__ == "__main__":
    model = psa.PhineasSentimentAnalyzer()
    model.train()
    comment = "This school has tried every way to make my son pay more money. We paid too much for that during my child's education here. so 😡😡😡 school . thank u !"
    print("Score: ", model.predict_score(comment))
    print("Negative") if model.predict_score(comment) < 0.5 else print("Positive")
