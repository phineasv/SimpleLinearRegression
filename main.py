import PhineasSentimentAnalyzer as psa

if __name__ == "__main__":
    model = psa.PhineasSentimentAnalyzer()
    model.train()
    comment = "This school has tried every way to make my son pay more money. We paid too much for that during my child's education here. so 😡😡😡 school . thank u !"
    print(model.predict_sentiment(comment))