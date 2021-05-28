import nltk
import streamlit as st
from PIL import Image

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from pickle import dump


tweets = pd.read_csv('sentiment_tweets3.csv')
tweets.drop(['Unnamed: 0'], axis=1, inplace=True)
tweets['label'].value_counts()

totalTweets = 8000 + 2314
trainIndex, testIndex = list(), list()
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.98:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = tweets.iloc[trainIndex]
testData = tweets.iloc[testIndex]

def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

class TweetClassifier(object):
    def __init__(self, trainData, method='tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                           len(list(self.tf_depressive.keys())))
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                       len(list(self.tf_positive.keys())))
        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.depressive_tweets + self.positive_tweets
        self.depressive_words = 0
        self.positive_words = 0
        self.tf_depressive = dict()
        self.tf_positive = dict()
        self.idf_depressive = dict()
        self.idf_positive = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets.iloc[i])
            count = list()
            for word in message_processed:
                if self.labels.iloc[i]:
                    self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                    self.depressive_words += 1
                else:
                    self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                    self.positive_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels.iloc[i]:
                    self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                else:
                    self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        self.sum_tf_idf_depressive = 0

        self.sum_tf_idf_positive = 0
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word]) * log(
                (self.depressive_tweets + self.positive_tweets) \
                / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
            self.sum_tf_idf_depressive += self.prob_depressive[word]
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (
                        self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                                      / (self.idf_depressive.get(word, 0) +
                                                                         self.idf_positive[word]))
            self.sum_tf_idf_positive += self.prob_positive[word]
        for word in self.tf_positive:
            self.prob_positive[word] = (self.prob_positive[word] + 1) / (
                        self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))

        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

    def classify(self, processed_message):
        pDepressive, pPositive = 0, 0
        for word in processed_message:
            if word in self.prob_depressive:
                pDepressive += log(self.prob_depressive[word])
            else:
                if self.method == 'tf-idf':
                    pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                else:
                    pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
            if word in self.prob_positive:
                pPositive += log(self.prob_positive[word])
            else:
                if self.method == 'tf-idf':
                    pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
                else:
                    pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
            pDepressive += log(self.prob_depressive_tweet)
            pPositive += log(self.prob_positive_tweet)
        return pDepressive >= pPositive

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
        true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
        false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
        false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)

    return accuracy

sc_tf_idf = TweetClassifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['message'])
metrics(testData['label'], preds_tf_idf)

bow = TweetClassifier(trainData, 'bow')
bow.train()



# pm = process_message("depression: i'm always here for u, darling")
# print(pm)
# print(sc_tf_idf.classify(pm))

# def accuracy():
#     return metrics(testData['label'], preds_tf_idf)
#
#
# def predict_tweet(message):
#     pm = process_message(message)
#     print(pm)
#     return sc_tf_idf.classify(pm)
#
# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
#
# def start():
#     st.title("Tweet Classifier ML app")
#     html_temp = """
#     <div style="background-color:black;padding:15px">
#     <h2 style="color:white";text-align:center;">Naive Bayes's  Theorem</h2>
#     </div>
#     """
#
#     st.markdown(html_temp,unsafe_allow_html=True)
#     load_css('style.css')
#
#     message = st.text_input("Enter Tweet")
#     if st.button("Predict"):
#         result = predict_tweet(message)
#         if result == False:
#             prediction = 'Non-Depressive Tweet'
#             st.success('Tweet was classified as {}'.format(prediction))
#             acc = accuracy()
#             st.success('Accuracy : {}'.format(acc))
#         elif result == True:
#             prediction = 'Depressive Tweet'
#             st.write("AASRA")
#             st.write("http://www.aasra.info")
#             st.write("Phone: 91-22-27546669")
#             st.write("Phone: 91-22-27546667")
#             st.write("Email: aasrahelpline@yahoo.com")
#             c_img = 'dep.jpg'
#             st.success('Tweet was classified as {}'.format(prediction))
#             accc = accuracy()
#             st.success('Metrics : {}'.format(accc))
#         else:
#             st.write("Failed to detect the tweet")
#
#         st.text("Shailesh Sachan")
#         st.text("Mujtaba Sayyed")
#         st.text("Taha Khan")
#
# start()
