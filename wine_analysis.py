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
from sklearn.feature_extraction import text
from sklearn.neighbors import LSHForest
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split


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
    
client = MongoClient()

#wine_info = client.wines.wine_info

# pull all reviews in without id
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


#cgt14 = client.wines.consolidated_gt14_wines
#wines = cgt14.find({}, {'_id':0})
#
##reviews = [review['review/text'] for review in wines]
#
##tfidf2, vect2 = vectorize_text(reviews)
#svd = TruncatedSVD(n_components=2)
#X_reviews = svd.fit(tfidf2).transform(tfidf2)
#print 'Shape of resulting principal components: {0}'.format(X_reviews.shape)
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
#rose = ['Ros&#233; Blend']
#
#color = []
#
#for variant in df['wine/variant'].values:
#    if variant in reds:
#        color.append('r')
#    elif variant in white:
#        color.append('y')
#    #elif variant in rose:
#    #    color.append('m')
#    else:
#        color.append('w')
#        
#plt.scatter(X_reviews[:,0], X_reviews[:,1], c=color)
#
#
#X = [(array, color) for (array, color) in zip(X_reviews, color) if color in ['r','y']]#,'m']]
#
#y = []
#
#for wine in X:
#    if wine[1] == 'r':
#        y.append(0)
#    else:
#        #wine[1] == 'y':
#        y.append(1)
#    #else:
#    #    y.append(2)
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
#svm = LinearSVC(penalty='l2').fit(X_train, y_train)#, dual=False, multi_class='ovr')
#y_pred = svm.predict(X_test)
#
#print metrics.classification_report(y_test, y_pred, target_names = ['class 0', 'class 1'])#, 'class 2'])
#
#cv = ShuffleSplit(len(X), n_iter=7, test_size=0.30, random_state=0)
#cv_accuracy = cross_validation.cross_val_score(svm, X, y, scoring='accuracy', cv=cv)
#print 'Cross Validation Accuracy: {0:0.2f} ({1:0.2f})'.format(cv_accuracy.mean(), cv_accuracy.std())

#classified_wines = svm.predict(X_reviews)
#red_reviews = X_reviews[classified_wines == 0]

#r_reviews = []
#
#for i in range(len(classified_wines)):
#    if classified_wines[i] == 0:
#        r_reviews.append(reviews[i])


#red_corpus_collection = client.wines.red_corpus_collection
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
#
#for review in r_reviews:
#    
#    red_cursor = cgt14.find({'review/text': review})
#    red_review = red_cursor.next()
#
#    words = []
#    sentences = sent_tokenize(red_review['review/text'].lower())
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
#    red_corpus_collection.insert({
#        "wine/name": red_review["wine/name"],
#        "wine/variant": red_review["wine/variant"],
#        "review/points": red_review['review/points'],
#        "review/text": red_review["review/text"],
#        "words": nouns
#    })
            
    
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


#dictionary_path = "models/dictionary.dict"
#corpus_path = "models/corpus.lda-c"
#lda_num_topics = 20
#lda_model_path = "models/lda_model_50_topics.lda"
#
#red_collection = client.wines.red_corpus_collection
#reviews_cursor = red_collection.find()
#
#dictionary = Dictionary(reviews_cursor, dictionary_path).build()
#Corpus(reviews_cursor, dictionary, corpus_path).serialize()
#Train.run(lda_model_path, corpus_path, lda_num_topics, dictionary)
#
##dictionary_path = "models/dictionary.dict"
##corpus_path = "models/corpus.lda-c"
##lda_num_topics = 50
##lda_model_path = "models/lda_model_50_topics.lda"
#
#dictionary = corpora.Dictionary.load(dictionary_path)
#corpus = corpora.BleiCorpus(corpus_path)
#lda = LdaModel.load(lda_model_path)
#
#i = 0
#for topic in lda.show_topics(num_topics=lda_num_topics):
#    print '#' + str(i) + ': ' + topic
#    i += 1
