# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:59:58 2015

@author: josephdziados
"""

from pymongo import MongoClient
import string
import pickle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.corpora import BleiCorpus
from gensim.models import LdaModel
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag
from operator import itemgetter
from collections import Counter
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction import text
from sklearn.neighbors import LSHForest
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

            
class Corpus(object):
    def __init__(self, cursor, reviews_dictionary, corpus_path):
        self.cursor = cursor
        self.reviews_dictionary = reviews_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
        self.cursor.rewind()
        for review in self.cursor:
            yield self.reviews_dictionary.doc2bow(review["words"])

    def serialize(self):
        BleiCorpus.serialize(self.corpus_path, self, id2word=self.reviews_dictionary)

        return self


class Dictionary(object):
    def __init__(self, cursor, dictionary_path):
        self.cursor = cursor
        self.dictionary_path = dictionary_path

    def build(self):
        self.cursor.rewind()
        dictionary = corpora.Dictionary(review["words"] for review in self.cursor)
        dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)

        return dictionary


class Train:
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        corpus = corpora.BleiCorpus(corpus_path)
        lda = LdaModel(corpus, num_topics=num_topics, id2word=id2word)
        lda.save(lda_model_path)

        return lda


def consolidate_wines(dataframe):
    
    """
    Takes in a dataframe and groups by the wine name; creates lists containing the wine name, all of its reviews 
    concatenated into one string, the varietal, the year, and the average review points
    """
    
    review_text = []
    avg_review_points = []
    wine_name = []
    wine_varietal = []
    wine_year = []


    for wine, iterables in dataframe.groupby('wine/name'):
        
        if len(set(iterables['review/text'].values)) > 14:

            # append wine name
            wine_name.append(wine)
    
            # combine all wine reviews in one string and append
            review_text.append(' '.join(set(iterables['review/text'].values)))
    
            # append wine varietal
            wine_varietal.append(iterables['wine/variant'].values[0])
    
            # append wine year
            wine_year.append(iterables['wine/year'].values[0])
    
            # append average review score
            avg_review_points.append(np.nanmean(iterables['review/points'].values))   
        
    keys = ['wine/name', 'wine/variant', 'review/text', 'review/points', 'wine/year']
    
    individual_wines = zip(wine_name, wine_varietal, review_text, avg_review_points, wine_year)
        
    return [dict(zip(keys, wine)) for wine in individual_wines]
    
 
def vectorize_text(x):#, tokenizer):
    
    added_stop_words = ['wine', 'nose', 'br', 'years', 'time', 'hour', 'bottle', '\' \'', '\'s', '\'m', 'n\'t', 'i', 'ca', \
      'come', 'came', 'did', 'just', 'wines', 'best', 'winery', '``', '\'\'', '\'ve', '...', 'winemaker', 'qpr']
    added_stop_words.extend([str(i) for i in range(2012)])
    added_stop_words.extend(string.punctuation)  
    stop_words = text.ENGLISH_STOP_WORDS.union(added_stop_words)

    vectorizer = text.TfidfVectorizer(min_df=1, stop_words=stop_words, ngram_range=(1,2))
    tfidf_reviews = vectorizer.fit_transform(x)
    print 'Shape of resulting tf-idf vectors: {0}'.format(tfidf_reviews.shape)
    
    return tfidf_reviews, vectorizer
    
    
def n_component_pca(n, tfidf_reviews):
    svd = TruncatedSVD(n_components=n)
    reviews = svd.fit(tfidf_reviews).transform(tfidf_reviews)
    print 'Shape of resulting principal components: {0}'.format(reviews.shape)
    return svd, reviews
    
    
client = MongoClient()

#wine_info = client.wines.wine_info
#
### pull all reviews in without id
#all_reviews = wine_info.find({}, {'_id': 0})
#print 'Total number of reviews: {0}'.format(all_reviews.count())
#
## create dataframe and replace 'N/A's in the review points column with NaN's in order to take averages on this column
#all_df = pd.DataFrame(list(all_reviews))
#all_df.replace({'review/points': {'N/A': np.nan}}, inplace=True)
#all_df['review/points'] = all_df['review/points'].astype(float)
#
## call function to get a list of wine dictionaries to send into mongo
#cons_wines = consolidate_wines(all_df)
#
## create mongo collection to put wines
#consolidated_gt14_wines = client.wines.consolidated_gt14_wines
## loop through list and add wine to collection
#for wine in cons_wines:
#    consolidated_gt14_wines.save(wine)


## PULL IN WINES HAVING 15 OR MORE REVIEWS THAT ARE OF YEAR 1990 OR OLDER
#cgt14 = client.wines.consolidated_gt14_wines
#wine_cursor = cgt14.find({'$and': [{'wine/year': {'$gt':'1990'}}, {'wine/year': {'$ne': 'N/A'}}]}, {'_id': 0})
#wines = [wine for wine in wine_cursor]

## PRINCIPAL COMPONENT ANALYSIS, VISUALIZATION, AND LINEAR SVM CLASSIFICATION
#reviews = [wine['review/text'] for wine in wines]
#tfidf_all, vect_all = vectorize_text(reviews)

#svd_2, reviews_2 = n_component_pca(2, tfidf_all)
#
#df = pd.DataFrame(list(wines))
#
#reds = ['Cabernet Sauvignon', 'Merlot', 'Cabernet Franc', 'Malbec', 'Shiraz, Syrah', 'Syrah', 'Red Blend', 'Grenache',
#        'Mourvedre', 'Rhone Red Blend', 'Pinot Noir', 'Gamay', 'Zinfandel', 'Petite Sirah', 'Tempranillo', 'Nebbiolo',
#        'Dolcetto', 'Nero d\'Avola', 'Primitivo', 'Barbera', 'Sangiovese', 'Carmenere', 'Pinotage', 'Red Bordeaux Blend', \
#       ]
#
#white = ['Chardonnay', 'Riesling', 'Sauvignon Blanc', 'White Blend', 'Champagne Blend', 'Cabernet Franc', \
#        'Gew&#252;rztraminer', 'Viognier', 'Chenin Blanc', 'Pinot Gris']
#
#color = []
#
#for variant in df['wine/variant'].values:
#    if variant in reds:
#        color.append('r')
#    elif variant in white:
#        color.append('y')
#    else:
#        color.append('w')
# 
### LINEAR SVM CLASSIFICATION 
#X = [(array, color) for (array, color) in zip(reviews_2, color) if color in ['r','y']]
#y = []
#
#for wine in X:
#    if wine[1] == 'r':
#        y.append(0)
#    else:
#        y.append(1)
#
#X = [array for (array, color) in X]
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
#print 'Length of X train set: {0}'.format(len(X_train))
#print 'Length of y train set: {0}'.format(len(y_train))
#print 'Length of X test set: {0}'.format(len(X_test))
#print 'Length of y test set: {0}'.format(len(y_test))
#
#svm = LinearSVC(penalty='l2').fit(X_train, y_train)
#y_pred = svm.predict(X_test)
#
#print metrics.classification_report(y_test, y_pred, target_names = ['class 0', 'class 1'])
#
#cv = ShuffleSplit(len(X), n_iter=7, test_size=0.30, random_state=0)
#cv_accuracy = cross_validation.cross_val_score(svm, X, y, scoring='accuracy', cv=cv)
#print 'Cross Validation Accuracy: {0:0.2f} ({1:0.2f})'.format(cv_accuracy.mean(), cv_accuracy.std())
#
#classified_wines = svm.predict(reviews_2)

#color = []
#for num in classified_wines:
#    if num == 0:
#        color.append('r')
#    else:
#        color.append('y')

#plt.figure(figsize=(10,10))       
#plt.scatter(reviews_2[:,0], reviews_2[:,1], c=color)
#plt.xlabel('PCA 1 - Fruit, Finish, Dark, Tannins, Red, Cherry, Black')
#plt.ylabel('PCA 2 - Citrus, Lemon, Apple, Yellow, Pear, Crisp, Honey')
#plt.title('Principal Component Analysis on Wines')

##LOCALITY SENSITIVE HASHING - NEAREST NEIGHBORS SEARCH 

# Actual analysis done with 25 components
#svd_all, all_reviews = n_component_pca(25, tfidf_all)
#
#lshf = LSHForest(n_neighbors = 5)
#lshf.fit(all_reviews)
##
#vec_reviews = vect_all.transform(['pair with shrimp, vanilla, oak, pineapple, butter'])
#svd_reviews = svd_all.transform(vec_reviews)
#neighbors = lshf.kneighbors(svd_reviews)
#
#for i in neighbors[1][0]:
#    print wines[i]
#    print

#joblib.dump(vect_all, 'vect_all.pkl')
#joblib.dump(svd_all, 'svd_all.pkl')
#pickle.dump(lshf, open("lshf_model.pkl", "w"))


#w_reviews = []
#r_reviews = []
#
#for i in range(len(classified_wines)):
#    if classified_wines[i] == 0:
#        r_reviews.append(reviews[i])
#    if classified_wines[i] == 1:
#        w_reviews.append(reviews[i])
 
#reds = client.wines.red_corpus_collection
#red_cursor = reds.find({}, {'_id': 0})
#red_reviews = [review['review/text'] for review in red_cursor]
#red_tfidf, red_vect = vectorize_text(red_reviews)
#svd25 = TruncatedSVD(n_components=25)
#red_svd = svd25.fit(red_tfidf).transform(red_tfidf)
#print 'Shape of resulting principal components: {0}'.format(red_svd.shape)
#
#store_pickles('svd25.pkl', svd25)
#store_pickles('red_vec.pkl', red_vect)
#store_pickles('red_svd_reviews.pkl', red_svd)
 
#whites = client.wines.white_corpus_collection
#white_cursor = whites.find({}, {'_id': 0})
#white_reviews = [review['review/text'] for review in white_cursor]
#white_tfidf, white_vect = vectorize_text(white_reviews)
#svd25_white = TruncatedSVD(n_components=25)
#white_svd = svd25_white.fit(white_tfidf).transform(white_tfidf)
#print 'Shape of resulting principal components: {0}'.format(white_svd.shape)
#
#store_pickles('svd25_white.pkl', svd25_white)
#store_pickles('white_vec.pkl', white_vect)
#store_pickles('white_svd_reviews.pkl', white_svd)





## TOPIC MODELING
#red_corpus_collection = client.wines.red_corpus_collection
#white_corpus_collection = client.wines.white_corpus_collection
#
#def load_stopwords():
#    more_stopwords = ['\'s', 'wine', 'nose', 'br', 'drink', 'year', 'time', 'day', 'hour', 'wines', 'winery' \
#    'glass', 'drank']
#    stopwords = {}
#    with open('stopwords.txt', 'rU') as f:
#        for line in f:
#            stopwords[line.strip()] = 1
#        for char in string.punctuation:
#            stopwords[char] = 1
#    for more_words in more_stopwords:
#        stopwords[more_words] = 1
#
#    return stopwords
#    
#
#stopwords = load_stopwords()

#for review in w_reviews:
#    
#    white_cursor = cgt14.find({'review/text': review})
#    white_review = white_cursor.next()
#
#    words = []
#    sentences = sent_tokenize(white_review['review/text'].lower())
#
#    for sentence in sentences:
#        tokens = word_tokenize(sentence)
#        review_text = [word for word in tokens if word not in stopwords]
#        tagged_text = pos_tag(review_text)
#
#        for word, tag in tagged_text:
#            words.append({"word": word, "pos": tag})
#    
#    lem= WordNetLemmatizer()
#
#    nouns = []
#    words = [word['word'] for word in words if word['pos'] in ["NN", "JJ"]]
#
#    for word in words:
#        nouns.append(lem.lemmatize(word))
#        
#    white_corpus_collection.insert({
#        "wine/name": white_review["wine/name"],
#        "wine/variant": white_review["wine/variant"],
#        "review/points": white_review['review/points'],
#        "review/text": white_review["review/text"],
#        "words": nouns
#    })


#r_dictionary_path = "models/r_60_dictionary.dict"
#w_dictionary_path = "models/w_60_dictionary.dict"
#r_corpus_path = "models/r_60_corpus.lda-c"
#w_corpus_path = "models/w_60_corpus.lda-c"
#lda_num_topics = 60
#r_lda_model_path = "models/r_lda_model_60_topics.lda"
#w_lda_model_path = "models/w_lda_model_60_topics.lda"

#white_collection = client.wines.white_corpus_collection
#w_reviews_cursor = white_collection.find()
#red_collection = client.wines.red_corpus_collection
#r_reviews_cursor = red_collection.find()

#r_dictionary = Dictionary(r_reviews_cursor, r_dictionary_path).build()
#w_dictionary = Dictionary(w_reviews_cursor, w_dictionary_path).build()
#Corpus(r_reviews_cursor, r_dictionary, r_corpus_path).serialize()
#Corpus(w_reviews_cursor, w_dictionary, w_corpus_path).serialize()
#Train.run(r_lda_model_path, r_corpus_path, lda_num_topics, r_dictionary)
#Train.run(w_lda_model_path, w_corpus_path, lda_num_topics, w_dictionary)

#r_dictionary_path = "models/r_60_dictionary.dict"
#w_dictionary_path = "models/w_60_dictionary.dict"
#r_corpus_path = "models/r_60_corpus.lda-c"
#w_corpus_path = "models/w_60_corpus.lda-c"
#lda_num_topics = 60
#r_lda_model_path = "models/r_lda_model_60_topics.lda"
#w_lda_model_path = "models/w_lda_model_60_topics.lda"

#dictionary = corpora.Dictionary.load(r_dictionary_path)
#corpus = corpora.BleiCorpus(r_corpus_path)
#lda = LdaModel.load(r_lda_model_path)
#dictionary = corpora.Dictionary.load(w_dictionary_path)
#corpus = corpora.BleiCorpus(w_corpus_path)
#lda = LdaModel.load(w_lda_model_path)

#i = 0
#for topic in lda.show_topics(num_topics=lda_num_topics):
#    print '#' + str(i) + ': ' + topic
#    i += 1

#wine_prices = pickle.load(open('wine_com_prices.pkl', 'r'))

