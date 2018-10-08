# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:22:29 2018

@author: yuanyu
"""

from pyspark import SparkContext

sc = SparkContext("local[4]", "First Spark App")
data = sc.textFile('UserPurchaseHistory.csv').map(lambda line: line.split(",")).map(lambda record: (record[0], record[1], record[2]))
numPurchase = data.count()
uniqueUsers = data.map(lambda record: record[0]).distinct().count()
totalRevenue = data.map(lambda record: float(record[2])).sum()

products = data.map(lambda record: (record[1], 1.0)).reduceByKey(lambda a, b: a + b).collect()
mostPopular = sorted(products, key=lambda x: x[1], reverse=True)[0]

print('Total Purchase:', numPurchase)
print('Unique users:', uniqueUsers)
print('Total revenue:', totalRevenue)
print('Most popular product:', mostPopular[0], 'with', mostPopular[1], 'purchases')
sc.stop()