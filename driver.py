
# coding: utf-8

# In[1]:

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier


# In[2]:

loc_imdb_tr = 'imdb_tr.csv'
# loc_stopwords = '/Users/michael/Documents/ai/proj-wk11/stopwords.en.txt'
loc_testcsv = '../resource/asnlib/public/imdb_te.csv'


# In[4]:

df = pd.DataFrame(columns=['text', 'polarity'])
directories = {1:'../resource/asnlib/public/aclImdb/train/pos/',
             0:'../resource/asnlib/public/aclImdb/train/neg/'}
i = 0
for x in directories.keys():
    for filename in os.listdir(directories[x]):
        if filename.endswith(".txt"):
            with open(directories[x]+filename) as infile:
                df.loc[i] = [infile.read(), int(x)]
                i += 1


# In[5]:

df.to_csv(loc_imdb_tr)


# In[7]:

# words = pd.Series([word for line in df.text.values for word in line.lower().split()]).value_counts()
# stopwords = []
# with open(loc_stopwords) as file:
#     for line in file:
#         line = line.strip() #or someother preprocessing
#         stopwords.append(line)


# In[16]:

cv_unigram = CountVectorizer(ngram_range=(1,1), stop_words='english')
cv_bigram  = CountVectorizer(ngram_range=(1,2), stop_words='english')
tdidf_unigram = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
tdidf_bigram  = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
dtm_cv_unigram = cv_unigram.fit_transform(df.text)
dtm_cv_bigram  = cv_bigram.fit_transform(df.text)
dtm_tdidf_unigram = tdidf_unigram.fit_transform(df.text)
dtm_tdidf_bigram  = tdidf_bigram.fit_transform(df.text)


# In[17]:

feat_cvu = cv_unigram.vocabulary_
feat_cvb = cv_bigram.vocabulary_
feat_tdu = tdidf_unigram.vocabulary_
feat_tdb = tdidf_bigram.vocabulary_


# In[18]:

testdf = pd.read_csv(loc_testcsv)
t_cv_unigram = CountVectorizer(ngram_range=(1,1), stop_words='english', vocabulary = feat_cvu, decode_error='ignore')
t_cv_bigram  = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary = feat_cvb, decode_error='ignore')
t_tdidf_unigram = TfidfVectorizer(ngram_range=(1,1), stop_words='english', vocabulary = feat_tdu, decode_error='ignore')
t_tdidf_bigram  = TfidfVectorizer(ngram_range=(1,2), stop_words='english', vocabulary = feat_tdb, decode_error='ignore')
t_dtm_cv_unigram = t_cv_unigram.fit_transform(testdf.text)
t_dtm_cv_bigram  = t_cv_bigram.fit_transform(testdf.text)
t_dtm_tdidf_unigram = t_tdidf_unigram.fit_transform(testdf.text)
t_dtm_tdidf_bigram  = t_tdidf_bigram.fit_transform(testdf.text)


# In[26]:

clf = SGDClassifier(loss="hinge", penalty="l1")
dtms = [dtm_cv_unigram, dtm_cv_bigram,dtm_tdidf_unigram, dtm_tdidf_bigram]
t_dtms = [t_dtm_cv_unigram, t_dtm_cv_bigram,t_dtm_tdidf_unigram, t_dtm_tdidf_bigram]
dtm_names = ['CV Unigram', 'CV Bigram', 'TDIDF Unigram', 'TDIDF Bigram']
output_names = ['unigram.output.txt','bigram.output.txt','unigramtfidf.output.txt','bigramtfidf.output.txt']
for dtm in range(4):
    clf.fit(dtms[dtm],df.polarity)
#     print "Accuracy score for {} is {}".format(dtm_names[dtm], clf.score(dtms[dtm],df.polarity))
    predictions = clf.predict(t_dtms[dtm]).tolist()
    thefile = open(output_names[dtm], 'w')
    for prediction in predictions:
        thefile.write("%s\n" % int(prediction))

