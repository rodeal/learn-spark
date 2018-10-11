# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:23:07 2018

@author: yuanyu
"""

from pyspark import SparkContext
import matplotlib.pyplot as plt

def convert_year(x):
    
    try:
        return int(x[-4:])
    except:
        return 1900
    
sc = SparkContext("local[4]", "First Spark App")

movie_data = sc.textFile('../data/ml-100k/u.item')
movie_fields = movie_data.map(lambda lines: lines.split('|'))
years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x[-4:]))
years_filtered = years.filter(lambda x: x != 1900)
movie_ages = years_filtered.map(lambda filtered: 1998 - filtered).countByValue()
values = list(movie_ages.values())
bins = list(movie_ages.keys())
plt.hist(values, bins=bins,color='lightblue', normed=True)

plt.bar(bins, values)
sc.stop()
