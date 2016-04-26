"""SimpleApp.py"""
import math
import re
import sys

from StringIO import StringIO
from datetime import datetime
from collections import namedtuple
from operator import add, itemgetter

# Note - SparkContext available as sc, HiveContext available as sqlCtx.
from pyspark import SparkContext
from pyspark import HiveContext
from pyspark.streaming import StreamingContext

sc = SparkContext(appName="PythonSentimentAnalysis")
sqlCtx = HiveContext(sc)

# Read in the word-sentiment list and create a static RDD from it
filenameAFINN = "/home/training/Assignment2/AFINN/AFINN-111.txt"

# map applies the lambda function (create a tuple of word and sentiment score) to every item of iterable
# within [ ] and returns a list of results. The dictionary is used here to be able to quickly lookup the
# sentiment score based on the key value 
afinn = dict(map(lambda (w, s): (w, int(s)), [ ws.strip().split('\t') for ws in open(filenameAFINN) ]))


# Read in the candidate mapping list and create a static dictionary from it
filenameCandidate = "file:///home/training/Assignment2/Candidates/Candidate Mapping.txt"

# map applies the lambda function
candidates = sc.textFile(filenameCandidate).map(lambda x: (x.strip().split(",")[0],x.strip().split(","))) \
				  	   .flatMapValues(lambda x:x).map(lambda y: (y[1],y[0])).distinct()


# word splitter pattern
pattern_split = re.compile(r"\W+")

# use sqlCtx to query the HIVE table
#tweets = sqlCtx.sql("select id, text, entities.user_mentions.name, entities.hashtags.text hashtag_text from twitteranalytics.tweets where id = '717160784451792900'")

#tweets = sqlCtx.sql("select id, text, entities.user_mentions.name, entities.hashtags.text hashtag_text from twitteranalytics.incremental_tweets")

tweets = sqlCtx.sql("select id, text, entities.user_mentions.name from twitteranalytics.incremental_tweets")


def sentiment(text):
 words = pattern_split.split(text.lower())
 sentiments = map(lambda word: afinn.get(word, 0), words)
 if sentiments:
  sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
  #sentiment = float(sum(sentiments))
 else:
  sentiment = 0
 return sentiment


#sentimentTuple = tweets.rdd.map(lambda r: [r.id, r.text, r.name, r.hashtag_text]) \
#			   .map(lambda r: [sentiment(r[1]),add(r[2],r[3])]) \
#			   .flatMapValues(lambda x: x) \
#			   .map(lambda y: (y[1],y[0])) \
#			   .reduceByKey(lambda x, y: x+y) \
#			   .sortByKey(ascending=True)


sentimentTuple = tweets.rdd.map(lambda r: [r.id, r.text, r.name]) \
			   .map(lambda r: [sentiment(r[1]),r[2]]) \
			   .flatMapValues(lambda x: x) \
			   .map(lambda y: (y[1],y[0])) \
			   .reduceByKey(lambda x, y: x+y) \
			   .sortByKey(ascending=True)

scoreDF = sentimentTuple.join(candidates) \
			.map(lambda (x,y): (y[1],y[0])) \
			.reduceByKey(lambda a,b: a+b) \
			.toDF()

scoreRenameDF = scoreDF.withColumnRenamed("_1","Candidate").withColumnRenamed("_2","Score")

sqlCtx.registerDataFrameAsTable(scoreRenameDF, "SCORE_TEMP")

sqlCtx.sql("INSERT OVERWRITE TABLE twitteranalytics.candidate_score \
	    SELECT Candidate, Score FROM SCORE_TEMP")

