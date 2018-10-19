# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:42:01 2018

@author: yuanyu
"""

import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

sc = SparkContext("local[4]", "First Spark App")

def to_implicit(x):
    if x <= 2:
        return 0
    else:
        return 1

#转换为Rating类
rawData = sc.textFile("../data/ml-100k/u.data")
rawRatings = rawData.map(lambda lines: lines.split('\t')[:3])
ratings = rawRatings.map(lambda fields: Rating(int(fields[0]), int(fields[1]), float(fields[2])))
implicit_ratings = ratings.map(lambda fields: Rating(fields[0], fields[1], to_implicit(fields[3])))
"""
模型参数：
rank: ALS模型中的因子个数，即低阶近似矩阵中的隐含特征个数 10-200
iterations: 迭代次数 10左右
lambda: 控制模型的正则化过程，防止过拟合；需要通过交叉验证来进行标定
"""
model = ALS.train(ratings, 50, 10, 0.01)

"""
模型返回：
MatrixFactorizationModel对象
用户因子RDD: model.userFeatures()
物品因子RDD: model.productFeatures()
"""
user_num = model.userFeatures().count()
movie_num = model.productFeatures().count()
print("用户数量:", user_num)
print("电影数量:", movie_num)