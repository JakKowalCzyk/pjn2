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

mails = pd.read_csv('spam.csv', encoding='latin-1', error_bad_lines=False)
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


def is_spam_email(email):
    bad_email = False
    bad_domain1 = False
    bad_domain2 = False
    with open('black_emails') as lines:
        for line in lines:
            if line == bad_email:
                bad_email = True

    domain = re.search(r'(@((\w+\.*)+))', email).group(2)
    with open('black_domains') as lines:
        for line in lines:
            if line == domain:
                bad_domain1 = True

    conn = http.client.HTTPSConnection("api.apility.net")
    headers = {
        'x-auth-token': "0e0669ea-e517-4009-bf6e-2fc43b0c70e2",
    }
    conn.request("GET", "/baddomain/" + domain, headers=headers)
    res = conn.getresponse()
    if res.code == 200:
        bad_domain2 = True

    bad_email_list = [bad_email, bad_domain1, bad_domain2]

    return sum(bad_email_list) / len(bad_email_list)


def is_spam_date(hour):
    hour_date = int(hour)
    if 22 < hour_date <= 24:
        return True
    if hour_date < 5:
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

print(sc_tf_idf.classify(process_message('I cant pick the phone right now. Pls send a message')))

print(sc_tf_idf.classify(process_message('Congratulations ur awarded $500 ')))

print(sc_tf_idf.classify(process_message('Nowa super wiadomosc')))

print("Podaj wiadomosc")
msg = input()

print("Podaj adres email nadawcy")
email = read_input(
    "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*",
    input(), "Podaj adres email nadawcy\n")

print("Podaj godzine odebrania wiadomosci")
hour = read_input("^([01]?\d|2[0-4])$", input(), "Podaj godzine odebrania wiadomosci\n")

is_spam_msg = False
is_bad_domain = False
is_bad_date = False
is_bad_domain_percentege = 0
is_bad_domain_percent = 0
if msg:
    is_spam_msg = sc_tf_idf.classify(process_message(msg))

if email:
    is_bad_domain_percentege = is_spam_email(email)
    if is_bad_domain_percentege == 1:
        is_bad_domain_percent = is_bad_domain_percentege * 100
        is_bad_domain = True
    elif is_bad_domain_percentege == 0:
        is_bad_domain = False
    else:
        is_bad_domain_percent = is_bad_domain_percentege * 100

if hour:
    is_bad_date = is_spam_date(hour)

can_be_spam = sum([is_spam_msg, is_bad_domain, is_bad_date]) + is_bad_domain_percentege
if can_be_spam == 0:
    print("Ta wiadomosc to nie spam. Żaden ze wskaźników nic nie wykrył")

elif can_be_spam != 0:
    if email and hour and msg:
        if False not in (is_spam_msg, is_bad_domain, is_bad_date):
            print("Ta wiadomosc to na pewno spam. Wszystkie czynniki na to wskazują")
        elif False not in (is_spam_msg, is_bad_date) and is_bad_domain_percent > 50:
            print("Wysokie prawdopodobieństwo, że adres należy do spamera.")
            print("Prawdopodobienstwo spamu to: ", ((2 + is_bad_domain_percentege) / 3) * 100, "%")
        else:
            if not email:
                if False not in (is_spam_msg, is_bad_date):
                    print("To prawdopodobnie spam")
                    print("Prawdopodobienstwo spamu to: ", (2 / 3) * 100, "%")
                else:
                    print("Prawdopodobienstwo spamu to: ", (((is_spam_msg + is_bad_date) / 2) * 100), "%")
            else:
                print("Prawdopodobienstwo spamu to: ",
                      (((is_spam_msg + is_bad_date + is_bad_domain_percentege) / 3) * 100), "%")
    else:
        if not email:
            if False not in (is_spam_msg, is_bad_date):
                print("To prawdopodobnie spam")
                print("Prawdopodobienstwo spamu to: ", (2 / 3) * 100, "%")
            else:
                print("Prawdopodobienstwo spamu to: ", ((((is_spam_msg + is_bad_date) / 2) * 100), "%"))
        else:
            print("Prawdopodobienstwo spamu to: ", (((is_spam_msg + is_bad_date + is_bad_domain_percentege) / 3) * 100),
                  "%")

    print("Wiadomosc oznaczona jako spam = ", "Tak" if is_spam_msg else "Nie")
    print("Godzina oznaczona jako spam = ", "Tak" if is_bad_date else "Nie")
    print("Prawdopodobienstwo, że adres email należy do spamera = ", is_bad_domain_percent, "%")
