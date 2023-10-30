import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
#from pyspark.ml.feature import QuantileDiscretizer
from flask import Flask, request
import numpy as np
spark = SparkSession.builder.appName("movieRecommendationPySpark").getOrCreate()
ratings = (
    spark.read.csv(
            path="D:/bigdata/ml-latest-small/ratings.csv",
            sep=",",
            header=True,
            quote="",
            schema="userId INT, movieId INT, rating DOUBLE, timestamp INT, title string",
            )
.select("userId", "movieId", "rating", "title" )
.cache()
)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator 

als = ALS( userCol="userId", itemCol="movieId", ratingCol="rating")
(training_data, validation_data) = ratings.randomSplit([8.0, 2.0])
evaluator = RegressionEvaluator( metricName="rmse", labelCol="rating", predictionCol="prediction")
model= als.fit(training_data) 
predictions = model.transform(validation_data)

app = Flask(__name__)

@app.route('/')
def home(): 
        return ("welcome")

@app.route('/user/<usid>/<rt>')
def user(usid,rt):
            userFeature = model.userFactors.filter(f.col('id')== usid ).select(f.col('features')).rdd.flatMap(lambda x: x).collect()[0]            
            itemFeature = model.itemFactors.filter(f.col('id')== rt ).select(f.col('features')).rdd.flatMap(lambda x: x).collect()[0]
            h = np.dot(userFeature, itemFeature)
            a = str(h) 
            return ('Dự đoán xếp hạng của người dùng cho phim: '+a)

@app.route('/top/<usid>')
def top(usid):

        pr= predictions.filter(f.col('rating') >= 2 ).filter(f.col('userId')==usid).select(f.col('title')).take(10)
        b= str(pr)
    
        return ('top 10 movie đề xuất '+b)

if __name__ == "__main__":
   app.run()
