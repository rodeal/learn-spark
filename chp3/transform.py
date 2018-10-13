# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:53:30 2018

@author: yuany
"""

import findspark
findspark.init()
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np

def convert_year(x):
    
    try:
        return int(x[-4:])
    except:
        return 1900

sc = SparkContext("local[4]", "First Spark App")

movie_data = sc.textFile('../data/ml-100k/u.item')
movie_fields = movie_data.map(lambda lines: lines.split('|'))

years_pre_processed = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x)).collect()
years_pre_processed_array = np.array(years_pre_processed)

mean_year = np.mean(years_pre_processed_array[years_pre_processed_array != 1900])
median_year = np.median(years_pre_processed_array[years_pre_processed_array != 1900])
index_bad_data = np.where(years_pre_processed_array == 1900)[0][0]
years_pre_processed_array[index_bad_data] = median_year
print('Mean year of release: %d' %mean_year)
print('Median year of release: %d' %median_year)
print('Index of \'1900\' after assigning median: %s' %np.where(years_pre_processed_array == 1900)[0])
 
sc.stop()