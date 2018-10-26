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
import numpy as np
from numpy import linalg as LA

sc = SparkContext("local[4]", "First Spark App")

def to_implicit(x):
    if x <= 2:
        return 0
    else:
        return 1

def cosineSimilarity(vec1, vec2):
    return float((np.dot(vec1, vec2) / (LA.norm(vec1) * LA.norm(vec2))))
    
#转换为Rating类
rawData = sc.textFile("../data/ml-100k/u.data")
rawRatings = rawData.map(lambda lines: lines.split('\t')[:3])
ratings = rawRatings.map(lambda fields: Rating(int(fields[0]), int(fields[1]), float(fields[2])))
implicit_ratings = ratings.map(lambda fields: Rating(fields[0], fields[1], int(to_implicit(fields[2]))))
"""
模型参数：
rank: ALS模型中的因子个数，即低阶近似矩阵中的隐含特征个数 10-200
iterations: 迭代次数 10左右
lambda: 控制模型的正则化过程，防止过拟合；需要通过交叉验证来进行标定
classmethod train(ratings, rank, iterations=5, lambda_=0.01, blocks=-1,
                                                  nonnegative=False, seed=None)
classmethod trainImplicit(ratings, rank, iterations=5, lambda_=0.01, blocks=-1,
                                      alpha=0.01, nonnegative=False, seed=None)
"""
model = ALS.train(ratings, 50, 10, 0.01)
model_implicit = ALS.train(ratings, 50, 10)

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

"""
PART I 用户推荐
模型应用:
预测：model.predict()
输出某用户的前k个推荐：model.recomendProducts(user, num)
predict函数可以输入以(user, item)ID对类型的RDD
"""
predictedRating = model.predict(789, 123)
userId = 789
K = 10
topKRecs = model.recommendProducts(userId, K)
print(topKRecs)

"""
模型检验
"""
movies = sc.textFile("../data/ml-100k/u.item")
titles = movies.map(lambda lines: lines.split("|")[:2]).map(lambda items: (int(items[0]), items[1])).collectAsMap()
#titles
moviesForUser = ratings.keyBy(lambda x: x.user).lookup(789)
print(len(moviesForUser))
sorted_ratings = sc.parallelize(moviesForUser).sortBy(lambda x: x.rating, ascending=False).take(10)
top10movies = [(titles[rating.product], rating.rating) for rating in sorted_ratings ]
top10recomm = [(titles[rating.product], rating.rating) for rating in topKRecs]

"""
Part II 物品推荐
"""
itemId = 567
itemFactor = model.productFeatures().lookup(itemId)[0].tolist()
itemVector = np.matrix(itemFactor).T
cosineSimilarity(itemVector.T, itemVector)
sims = model.productFeatures().map(lambda items: (items[0], cosineSimilarity(np.matrix(items[1].tolist()), itemVector)))
sortedSims = sims.top(K, key=lambda x: x[1])
sortedSims2 = sims.top(K+1, key=lambda x: x[1])
similar_titles = [(titles[x[0]], x[1]) for x in sortedSims2[1:11]]

