from concurrent.futures.process import _threads_wakeups
from operator import countOf
import requests
from bs4  import BeautifulSoup
from nltk import tokenize

import re, string
import time
from apscheduler.schedulers.blocking import BlockingScheduler
import re
import string
import pandas as pd
import numpy as np
#import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.tree import DecisionTreeClassifier

# Random seed for consistency
np.random.seed(500)

def train_data():
    # Uncomment line below to download nltk requirements
    # nltk.download('punkt')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    Encoder = LabelEncoder()
    Tfidf_vect = TfidfVectorizer()

    # Configuration
    DATA_LATIH = "./DataLatih.csv"

    # Open Datasets
    datasets = pd.read_csv(DATA_LATIH)
    print(datasets["label"].value_counts())

    # Text Normalization using
    # PySastrawi(Word Stemmer for Bahasa Indonesia)
    lower   = [stemmer.stem(row.lower()) for row in datasets["narasi"]]
    vectors = [word_tokenize(element) for element in lower]
    labels  = datasets["label"]

    # Splitting Datasets for feeding to Machine Learning
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(
        vectors, labels, test_size=0.25, stratify=labels)

    # Encoder for Data Label
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y  = Encoder.fit_transform(Test_Y)

    # Create Tfidf Vector
    Tfidf_vect.fit(["".join(row) for row in lower])

    # Applying Tfidf for Training and Testing Features
    Train_X_Tfidf   = Tfidf_vect.transform([" ".join(row) for row in Train_X])
    Test_X_Tfidf    = Tfidf_vect.transform([" ".join(row) for row in Test_X])

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma="auto", verbose=True)
    SVM.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
    # Use accuracy_score function to get the accuracy
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)
    print(classification_report(Test_Y, predictions_SVM))
    #return SVM

    # Classifier - Algorithm - Logistic Regretion
    # fit the training dataset on the classifier
    LR = LogisticRegression()
    LR.fit(Train_X_Tfidf, Train_Y)
    #LogisticRegression()
    # Use accuracy_score function to get the accuracy
    LR.score(Test_X_Tfidf, Test_Y)
    prediksi_LR = LR.predict(Test_X_Tfidf)
    print ("Logistric Regretion score -> ", accuracy_score(prediksi_LR, Test_Y)*100)
    print(classification_report(Test_Y, prediksi_LR))

    # Classifier - Algorithm - Decision Tree Classification
    # fit the training dataset on the classifier
    DT = DecisionTreeClassifier()
    DT.fit(Train_X_Tfidf, Train_Y)
    #DecisionTreeClassifier()
    # Use accuracy_score function to get the accuracy
    DT.score(Test_X_Tfidf, Test_Y)
    prediksi_DT = DT.predict(Test_X_Tfidf)

    print ("Decission Tree score -> ", accuracy_score( Test_Y, prediksi_DT)*100)
    print(classification_report(Test_Y, prediksi_DT))

    # Classifier - Algorithm - Gradient Boosting Classifier
    # fit the training dataset on the classifier
    GBC = GradientBoostingClassifier(random_state=0)
    GBC.fit(Train_X_Tfidf, Train_Y)
    #GradientBoostingClassifier()
    # Use accuracy_score function to get the accuracy
    GBC.score(Test_X_Tfidf, Test_Y)
    prediksi_GBC = GBC.predict(Test_X_Tfidf)

    print ("Gradient Boosting Classifier score -> ", accuracy_score( Test_Y, prediksi_GBC)*100)
    print(classification_report(Test_Y, prediksi_GBC))

    # Classifier - Algorithm - Random Forest
    # fit the training dataset on the classifier
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(Train_X_Tfidf, Train_Y)

    # Use accuracy_score function to get the accuracy
    RFC.score(Test_X_Tfidf, Test_Y)
    prediksi_RFC = RFC.predict(Test_X_Tfidf)

    print ("Random Forest Classifier score -> ", accuracy_score( Test_Y, prediksi_RFC)*100)
    print(classification_report(Test_Y, prediksi_RFC))

def word_drop(text):
    remove = string.punctuation
    remove = remove.replace(".", "")
    remove = remove.replace('"', "")
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    #text = re.sub("\\W"," ",text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(remove), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)    
    return text

def cari_berita():
    from firebase import firebase
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    firebase = firebase.FirebaseApplication('https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

    url = 'https://www.kompas.com/'
    html        = requests.get(url)
    soup        = BeautifulSoup(html.content, 'lxml')

    populer     = soup.find('div', {'class', 'most__wrap clearfix'})
    isi_berita  = populer.find_all('div', {'class', 'most__list clearfix'})
    list_judul_berita = []
    for each in isi_berita:
        nomor = each.find('div',{'class','most__count'}).text
        judul_berita = each.find('h4',{'class','most__title'}).text
        list_judul_berita.append(judul_berita) 
        link_berita = each.a.get('href')+"?page=all"
        print(nomor)
        print(judul_berita)
        print(link_berita)
        print('')
        
    links = []
    for each in isi_berita:
        links.append(each.a.get('href'))

    print("Jumlah berita adalah {}".format(len(links)))   
    i = 0

    for link in links:
        i       = i + 1
        page    = requests.get(link)
        halaman_berita    = BeautifulSoup(page.content, 'lxml')
        Isi_Berita = halaman_berita.find('div',{'class','read__content'}).text

        berita = tokenize.sent_tokenize(Isi_Berita)
        beritanya = []
        for kalimat_berita in berita:
            if "KOMPAS.com - " in kalimat_berita: kalimat_berita = kalimat_berita.replace("KOMPAS.com - ","")
            if kalimat_berita.lower().find("baca juga:") >= 0 :
                kalimat_berita = ''         
                    
            if kalimat_berita.lower().find('dapatkan update berita pilihan') >= 0 :
                kalimat_berita = ''
                break

            #kalimat_berita = cleanedthings(kalimat_berita)
            beritanya.append(kalimat_berita)
            #print (kalimat_berita)
        
        kalimat_berita_bersih = word_drop(' '.join(beritanya))
        
        #cleanedthings(kalimat_berita_bersih)
        if "  " in  kalimat_berita_bersih:  kalimat_berita_bersih =  kalimat_berita_bersih.replace("  "," ") 
        
        firebase.post('/Berita', {'Judul':list_judul_berita[i-1],'Konten':kalimat_berita_bersih})
        print('======= berita ke-{} ======='.format(i))
        print(kalimat_berita_bersih)


train_data()
sched = BlockingScheduler(timezone="Asia/Jakarta")
sched.add_job(cari_berita, "interval", minutes=2)

sched.start()

    