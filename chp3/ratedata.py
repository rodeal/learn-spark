# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:12:37 2018

@author: yuanyu
"""

from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext("local[4]", "First Spark App")

rating_data = sc.textFile('../data/ml-100k/u.data').map(lambda line: line.split('\t'))
ratings = rating_data.map(lambda fields: int(fields[2]))
num_ratings = ratings.count()
num_users = rating_data.map(lambda fields: fields[0]).distinct().count()
num_movies = rating_data.map(lambda fields: fields[1]).distinct().count()
max_rating = ratings.reduce(lambda x, y: max(x, y))
min_rating = ratings.reduce(lambda x, y: min(x, y))
mean_rating = ratings.reduce(lambda x, y: x + y) / num_ratings
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings / num_users
ratings_per_movie = num_ratings / num_movies

print("Min rating: %d" %min_rating)
print("Max rating: %d" %max_rating)
print("Average rating: %2.2f" %mean_rating)
print("Average # of ratings per user: %2.2f" %ratings_per_user)
print("Average # of ratings per movie: %2.2f" %ratings_per_movie)

count_by_rating = ratings.countByValue()
x_axis = list(count_by_rating.keys())
y_axis = [float(c) for c in count_by_rating.values()]
y_axis_normed = y_axis / y_axis.sum()
pos = np.arange(len(x_axis))
width = 1.0

ax = plt.axes()
ax.set_xticks(pos + width / 2)
ax.set_xticklabels(x_axis)

plt.bar(pos, y_axis_normed, width, color='lightblue')
plt.xticks(rotation=30)
      
sc.stop()