# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:12:12 2018

@author: yuany
"""

import findspark
findspark.init()
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
import datetime
import re
from scipy import sparse as sp

sc = SparkContext("local[4]", "First Spark App")

def extract_datetime(ts):
    return datetime.datetime.fromtimestamp(ts)

def assign_tod(hr):
    times_of_day = {
        'morning' : range(7, 12),
        'lunch' : range(12, 14),
        'afternoon': range(14, 18),
        'evening': range(18, 23),
        'night': range(23, 7)
    }
    for key in times_of_day.keys():
        if hr in times_of_day[key]:
            return key
    return 'night'
    
def extract_title(raw):
    grps = re.search("\((\w+)\)", raw)
    if grps:
        return raw[:grps.start()].strip()
    else:
        return raw
   
def create_vector(terms, term_dict):
    num_terms = len(term_dict)
    x = sp.csc_matrix((1, num_terms))
    for t in terms:
        if t in term_dict:
            idx = term_dict[t]
            x[0, idx] = 1
    return x
    
user_data = sc.textFile('../data/ml-100k/u.user')
user_fields = user_data.map(lambda line: line.split('|'))

all_occupations = user_fields.map(lambda fields: fields[3]).distinct().collect()
all_occupations.sort()

"""
类别特征 1 of k 方法，或1-hot encoding
"""
idx = 0
all_occupations_dict = {}
for o in all_occupations:
    all_occupations_dict[o] = idx
    idx += 1
print("Encoding of 'doctor': %d" %all_occupations_dict['doctor'])
print("Encoding of 'programmer': %d" %all_occupations_dict['programmer'])
K = len(all_occupations_dict)
binary_x = np.zeros(K)
k_programmer = all_occupations_dict['programmer']
binary_x[k_programmer] = 1
print('Binary feature vector: %s' %binary_x)
print('Length of binary vector: %d' %K)

"""
派生特征
"""
rating_data = sc.textFile('../data/ml-100k/u.data').map(lambda line: line.split('\t'))
timestamps = rating_data.map(lambda fields: int(fields[3]))
hour_of_day = timestamps.map(lambda ts: extract_datetime(ts).hour)
hour_of_day.take(5)
time_of_day = hour_of_day.map(lambda hr: assign_tod(hr))
time_of_day.take(5)

"""
文本特征 bag of word
"""
movie_data = sc.textFile('../data/ml-100k/u.item')
movie_fields = movie_data.map(lambda lines: lines.split('|'))
raw_titles = movie_fields.map(lambda fields: fields[1])
#for raw_title in raw_titles.take(5):
#    print(extract_title(raw_title))
movie_titles = raw_titles.map(lambda m: extract_title(m))
title_terms = movie_titles.map(lambda t: t.split(' '))
print(title_terms.take(5))

all_terms = title_terms.flatMap(lambda x: x).distinct().collect()
idx = 0
all_terms_dict = {}
for term in all_terms:
    all_terms_dict[term] = idx
    idx += 1
print('Total number of terms: %d' %len(all_terms_dict))
print("Index of 'Dead': %d" %all_terms_dict['Dead'])
print("Index of term 'Rooms': %d" %all_terms_dict['Rooms'])

all_terms_dict2 = title_terms.flatMap(lambda x: x).distinct().zipWithIndex().collectAsMap()
print("Index of 'Dead': %d" %all_terms_dict2['Dead'])
print("Index of term 'Rooms': %d" %all_terms_dict2['Rooms'])

all_terms_bcast = sc.broadcast(all_terms_dict)
all_terms_bcast_value = all_terms_bcast.value
term_vectors = title_terms.map(lambda terms: create_vector(terms, all_terms_bcast_value))
term_vectors.take(5)
sc.stop()