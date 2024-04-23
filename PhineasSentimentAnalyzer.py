import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np
from tqdm import tqdm
import pandas as pd

class PhineasSentimentAnalyzer:

    def __init__(self):
        self.alpha = 0.15
        self.num_iterations = 1000
        self.pos_list, self.neg_list, self.stop_word_list = self.__build_pos_neg_list("./positive-words.txt", "./negative-words.txt", "./nltk.data/english.txt")
        self.training_data = pd.read_csv("./training_data.csv").drop('Unnamed: 0', axis = 1).iloc[:1000]
        self.freqs = self.__build_freqs(self.training_data.value, self.training_data.p)
        self.theta = [1,1,1,1,1,0]

    def __process_comment(self, comment):
        # Change any type to string before preprocessing
        comment = str(comment)

        # Removing URL link
        comment = re.sub(r'(https?://\S+)','', comment)

        # Only include number and letters
        regex = r"[\w]+"
        comment = " ".join(re.findall(regex, comment))

        # Removing only hastags from the words
        comment = re.sub(r'#','',comment)

        # Tokenize comments
        tokenizer = TweetTokenizer(preserve_case = False, reduce_len=True, strip_handles=True)
        comment = tokenizer.tokenize(comment)

        # Stopwords
        comment_clean = []
        for word in comment:
            if word not in self.stop_word_list and word not in string.punctuation:
                comment_clean.append(word)

        # Stemming
        stemmer = PorterStemmer()
        comment_stem = []
        for word in comment_clean:
            stem_word = stemmer.stem(word)
            comment_stem.append(stem_word)

        return comment_stem

    def __build_freqs(self, comments, labels):
        labels_list = np.squeeze(labels).tolist()
        freqs = {}

        for y, comment in tqdm(zip(labels_list, comments)):
            for word in self.__process_comment(comment): 
                pair = (word, y)
                freqs[pair] = freqs.get(pair, 0) + 1

        return freqs
    
    def __build_pos_neg_list(self, pos_list_path, neg_list_path, stopword_path):
        f_pos = open(pos_list_path, "r", encoding="ISO-8859-1")
        f_neg = open(neg_list_path, "r", encoding="ISO-8859-1")
        f_stop = open(stopword_path, "r", encoding="ISO-8859-1")
        neg_list = []
        pos_list = []
        stopword_list = []
        for x in f_pos:
            pos_list.append(x.strip())

        for x in f_neg:
            neg_list.append(x.strip())

        for x in f_stop:
            stopword_list.append(x.strip())
    
        return pos_list, neg_list, stopword_list
    
    def __find_sigmoid_func(self, z):
        '''
            Input:
                z: is the input (can be a scalar or an array)
            Output:
                h: the sigmoid of z
        '''
        h = 1/(1 + np.exp(-z))
        return h
    
    def __find_y_pred(self, X, theta):
        '''
            Input:
            X: matrix (m, n+1) for features and values
            theta: (n+1, 1) for weights for each features
            Output:
            h: y_pred
        '''
        y_pred = np.dot(X,theta)
        h = self.__find_sigmoid_func(y_pred)
        return h
    
    def __find_loss_func(self, y_pred, y):
        '''
            Input:
            y_pred: The y that is predicted
            y: The real y
            Output:
            l: Loss function
        '''
        res = -(y*np.log(y_pred)+(1-np.array(y))*np.log(1- np.array(y_pred))) 
        return res.sum()/res.size
    
    def __find_gradient(self, X, y, theta, alpha, num_iters):
        '''
            Input:
                X: matrix (m, n+1) for features and values
                y: (m, 1) for real values
                theta: (n+1, 1) for weights for each features
                alpha: learning rate
                num_iters: number of iterations
            Output:
                theta: (n+1, 1) for weights for each features
                y_pred: (m, 1) for predicted values
        '''
        # Gradient
        m = len(X)
        n = len(theta)
        for i in tqdm(range(0, num_iters)):
            y_pred = self.__find_y_pred(X, theta)
            # deltaCost_w = (1/m) * np.dot((y_pred - y).reshape(1,m),X)
            deltaCost_w = (1/m) * np.dot((y_pred - y),X)
            theta = theta - alpha*deltaCost_w
        # Find y_pred, loss for reporting
        # theta = theta.reshape(n,1)
        return theta, y_pred
    
    def __extract_features(self, comment, pos_list, neg_list, freqs):
        '''
            Input: 
            comment: the comment line
            freqs: frquency table
            Output: 
            x: vector of features
        '''
        comment = str(comment)
        comment_full_split = str(comment.lower()).split(" ")
        first_sec_pronouns = ["i", "me", "my", "mine", "you", "your", "yours"]
        x = np.zeros((1,6))
        # Check if there is no or not in sentence  
        x[0,0] = 1 if "no" in comment_full_split else 0

        # Count how many first and second pronouns
        for word in comment_full_split:
            if word in first_sec_pronouns:
                x[0,1] += 1 

        processed_comment = self.__process_comment(comment)

        for word in comment_full_split:
            # Number of positive lexicon
            if word in pos_list:
                x[0,2] += 1 

            # Number of negative lexicon
            if word in neg_list:
                x[0,3] += 1 

        for word in processed_comment:
            x[0,2] += freqs.get((word, 1), 0)
            x[0,3] += freqs.get((word, 0), 0)

        x[0,2] = 0 if x[0,2] == 0 else np.log(x[0,2])
        x[0,3] = 0 if x[0,3] == 0 else np.log(x[0,3])

        # Word count
        x[0,4] = np.log(len(comment_full_split))

        # Last column for b
        x[0,5] = 1
        return x
    
    def train(self):
        comments = self.training_data.value
        X = np.zeros((len(self.training_data), 6))
        for i in tqdm(range(len(self.training_data))):
            X[i,:] = self.__extract_features(comment=comments[i], pos_list=self.pos_list, neg_list=self.neg_list, freqs=self.freqs)
        y = self.training_data.p
        self.theta, y_pred = self.__find_gradient(X, y, self.theta, self.alpha, self.num_iterations)
        return
    
    def predict_score(self, comment):
        X = self.__extract_features(comment, self.pos_list, self.neg_list, self.freqs)
        y_pred = self.__find_y_pred(X, self.theta)
        return y_pred
    
    def predict_sentiment(self, comment):
        y_pred = self.predict_score(comment)
        return 1 if y_pred >= 0.5 else 0

    def extract(self, comment):
        return self.__extract_features(comment, self.pos_list, self.neg_list, self.freqs)
    

model = PhineasSentimentAnalyzer()
model.train()
sentence = "Finding a job is a difficult thing."
print("Score: ", model.predict_score(sentence))
print("Negative") if model.predict_score(sentence) < 0.5 else print("Positive")
model.extract(sentence)
