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
with open('prices.pkl', 'r') as prices:
    prices = pickle.load(prices)

client = MongoClient()

cgt14 = client.wines.consolidated_gt14_wines
wine_cursor = cgt14.find({'$and': [{'wine/year': {'$gt':'1990'}}, {'wine/year': {'$ne': 'N/A'}}]}, {'_id': 0})
wines = [wine for wine in wine_cursor]
food_list = ['asparagus', 'barbecue', 'beef', 'broccoli', 'sprouts', 'brisket', \
             'burrito', 'casserole', 'cheese', 'chicken', 'chili', 'greens', 'crab', \
             'dessert', 'fillet', 'fish', 'gyro', 'halibut', 'burger', 'hot dog', 'ice cream', 'kale', \
             'beans', 'lamb', 'lasagna', 'legumes', 'lentils', 'lobster', 'macaroni', \
             'potato', 'mozzarella', 'pizza', 'mussels', 'noodles', 'oyster', 'pasta', 'squash', \
             'pork', 'prime rib', 'pretzel', 'quinoa', 'ravioli', 'roast', \
             'salami', 'salmon', 'sausage', 'scallops', 'shrimp', 'soup', 'spaghetti', 'spareribs', 'spinach', \
             'peas', 'squid', 'steak', 'stir-fry', 'sushi', 'tapioca', 'teriyaki', 'turkey', 'turnip', 'tuna steak', \
             'veal', 'venison', 'yam', 'zucchini']

for wine in wines:
    rounded = round(wine['review/points'], 2)
    wine['review/points'] = rounded
    
#    if wine['wine/name'] in prices.keys():
#        wine['price'] = prices[wine['wine/name']]
#    else:
#        wine['price'] = 'Not Available'
    
    foods = []    
    for food in food_list:
        if food in wine['review/text']:
            foods.append(food)
    wine['food_pairing'] = ', '.join(foods)

             
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
    
    print data
    
    flavors1 = data['descriptions']
    flavors2 = data['pair_desc']
    
#    if '[' in flavors1:
#        key_words = ' '.join(flavors1.split(',')).lower()
#    else:
    key_words2 = ' '.join(flavors2).lower()
    key_words = re.findall('[A-Z][^A-Z]*', flavors1)
    key_words = ' '.join(key_words).lower()
    flavors = []
    flavors.append(key_words)

    vec_reviews = vect_all.transform(flavors)
    svd_reviews = svd_all.transform(vec_reviews)
    neighbors = lshf.kneighbors(svd_reviews)
    
    vec_reviews2 = vect_all.transform(key_words2)
    svd_reviews2 = svd_all.transform(vec_reviews2)
    neighbors2 = lshf.kneighbors(svd_reviews2)

    nearest_wines = []
    nearest_pairings = []
    for i in neighbors[1][0]:
        nearest_wines.append(wines[i])
    nearest_wines = sorted(nearest_wines, key=itemgetter('review/points'), reverse=True)
    
    for i in neighbors2[1][0]:
        nearest_pairings.append(wines[i])
    nearest_pairings = sorted(nearest_pairings, key=itemgetter('review/points'), reverse=True)

    results = jsonify({'wine': nearest_wines, 'pairing': nearest_pairings})
    
    return results