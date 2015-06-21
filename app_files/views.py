# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:57:51 2015

@author: josephdziados
"""
from flask import jsonify, render_template, request, Flask
import numpy as np
import pandas as pd
import pickle
import re
from app import app
from operator import itemgetter
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.neighbors import LSHForest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text

vect_all = joblib.load('vect_all.pkl')
svd_all = joblib.load('svd_all.pkl')
with open('lshf_model.pkl', 'r') as lshf:
    lshf = pickle.load(lshf)

client = MongoClient()

cgt14 = client.wines.consolidated_gt14_wines
wine_cursor = cgt14.find({'$and': [{'wine/year': {'$gt':'1990'}}, {'wine/year': {'$ne': 'N/A'}}]}, {'_id': 0})
wines = [wine for wine in wine_cursor]
for wine in wines:
    rounded = round(wine['review/points'], 2)
    wine['review/points'] = rounded

# Homepage
@app.route('/')
@app.route('/index')
def index():
    with open('index.html', 'r') as home:
        return home.read()
        
# Get an example and return its score
@app.route('/wines', methods=["POST"])
def find_wines():

    data = request.json
    key_words = data['descriptions']
    if '[' in key_words:
        key_words = ' '.join(key_words.split(',')).lower()
    else:
        key_words = re.findall('[A-Z][^A-Z]*', key_words)
        key_words = ' '.join(key_words).lower()
    flavors = []
    flavors.append(key_words)

    vec_reviews = vect_all.transform(flavors)
    svd_reviews = svd_all.transform(vec_reviews)
    neighbors = lshf.kneighbors(svd_reviews)

    nearest_wines = []
    for i in neighbors[1][0]:
        nearest_wines.append(wines[i])
    nearest_wines = sorted(nearest_wines, key=itemgetter('review/points'), reverse=True)

    return jsonify({'wine': nearest_wines})