import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from flask import Flask
import numpy as np
spark = SparkSession.builder.appName("movieRecommendationPySpark").getOrCreate()
ratings = (
    spark.read.csv(
            path="D:/bigdata/ml-latest-small/ratings.csv",
            sep=",",
            header=True,
            quote="",
            schema="userId INT, movieId INT, rating DOUBLE",
            )
.select("userId", "movieId", "rating")
.cache()
)
movies = (
    spark.read.csv(
            path="D:/bigdata/ml-latest-small/movies.csv",
            sep=",",
            header=True,
            quote="",
            schema="movieId INT,title string",
            )
.select("movieId", "title")
.cache()
) 

joins = ratings.join(movies, on=["movieId","movieId"], how="inner").select("userId", "movieId", "rating", "title")

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator 

als = ALS( userCol="userId", itemCol="movieId", ratingCol="rating")
(training_data, validation_data) = joins.randomSplit([8.0, 2.0])
evaluator = RegressionEvaluator( metricName="rmse", labelCol="rating", predictionCol="prediction")
model= als.fit(training_data) 
predictions = model.transform(validation_data);

app = Flask(__name__)

@app.route('/')
def home(): 
        return ("welcome")
@app.route('/user/<usid>/<mv>/<rt>')
def user(usid,mv,rt):
    ad = [(usid, mv, rt)]
    df = (spark.createDataFrame(ad, schema=["userId", "movieId", "rating"]).cache()) 
    jdf = ratings.union(df)
    jdf.write.mode("overwrite").saveAsTable("ratings")
    ratings.write.format("csv").mode("overwrite").save("ratings.csv")
    ns = df.filter(f.col('userId') == usid).select(f.col('userId')).take(1)
    n = ns[0]
    return ("welcome new user {}".format(n))
@app.route('/rating/<usid>/<mv>')
def rating(usid,mv):       
            userFeature = model.userFactors.filter(f.col('id') == usid ).select(f.col('features')).rdd.flatMap(lambda x: x).collect()[0]            
            itemFeature = model.itemFactors.filter(f.col('id') == mv ).select(f.col('features')).rdd.flatMap(lambda x: x).collect()[0]
            h = np.dot(userFeature, itemFeature)               
            a = h[0]         
            return ('rating của user cho phim: {}'.format(a))

@app.route('/top/<usid>')
def top(usid):
            
            pr = predictions.filter(f.col('rating') >= 2 ).filter(f.col('userId')==usid).select(f.col('title')).take(10)
            l = str(pr)
            return ('top 10 movie đề xuất '+l)

if __name__ == "__main__":
   app.run()
