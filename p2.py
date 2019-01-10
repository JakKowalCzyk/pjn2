from nltk.tokenize import word_tokenize
import http.client
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re

mails = pd.read_csv('spam.csv', encoding='latin-1')
mails.head()

mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
mails.head()

mails.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
mails.head()

mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})

mails.drop(['labels'], axis=1, inplace=True)

trainData = mails

trainData.reset_index(inplace=True)
trainData.drop(['index'], axis=1, inplace=True)

def process_message(message):
    message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    sw = stopwords.words('english')
    return [word for word in words if word not in sw]


class SpamClassifier(object):
    def __init__(self, trainData):
        self.mails, self.labels = trainData['message'], trainData['label']

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()

    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.mails[i])
            count = list()
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                              / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                    self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                            / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

def is_bad_domain(email):
    conn = http.client.HTTPSConnection("api.apility.net")
    headers = {
        'x-auth-token': "0e0669ea-e517-4009-bf6e-2fc43b0c70e2",
    }
    conn.request("GET", "/baddomain/" + email, headers=headers)
    res = conn.getresponse()
    if res.code == 200:
        return True
    return False


def is_spam_date(hour):
    hour_date = int(hour)
    if 22 < hour_date < 5:
        return True
    return False

def read_input(reg, name, field_name):
    if not name:
        return name
    m = re.match(reg, name)
    if m:
        return name
    else:
        print("zly format")
        return read_input(reg, input(field_name), field_name)

sc_tf_idf = SpamClassifier(trainData)
sc_tf_idf.train()

print(sc_tf_idf.classify(process_message('I cant pick the phone right now. Pls send a message')))

print(sc_tf_idf.classify(process_message('Congratulations ur awarded $500 ')))

print(sc_tf_idf.classify(process_message('Nowa super wiadomosc')))

print("Podaj wiadomosc\n")
msg = input()

print("Podaj adres email nadawcy\n")
email = read_input("[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*", input(), "Podaj adres email nadawcy\n")

print("Podaj godzine odebrania wiadomosci\n")
hour = read_input("^([01]?\d|2[0-4])$", input(), "Podaj godzine odebrania wiadomosci\n")

if msg:
    is_spam_msg = sc_tf_idf.classify(process_message(msg))

if email:
    is_bad_domain = is_bad_domain(re.search(r'(@(\w+\.*)+)', email).group(1))

if hour:
    is_bad_date = is_spam_date(hour)
