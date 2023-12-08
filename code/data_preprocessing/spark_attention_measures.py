import os
import sys
sys.path.append('')
import hdfs
import time
import datetime
import subprocess
import numpy as np
import ujson as json
from itertools import chain


os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
import findspark
findspark.init()
findspark.os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
findspark.os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.ml.feature import Bucketizer
from pyspark.sql import SparkSession, Window
from pyspark import SparkContext, SparkConf

from pipeline_helper import read_panel_data
from pipeline_helper import reindex
from pipeline_helper import delete_spark_staging_dir
from pipeline_helper import agg_spark_output


DEBUG = False

PANEL_TWEETS_LOC = "hdfs://tweets/"
TWEETS_SCHEMA = "tweet_schema.json"

if DEBUG:
    PANEL_TWEETS_LOC = "hdfs://panel_tweets"


out_dir = "hdfs://out_dir"



def start_spark(name, debug = False):

    delete_spark_staging_dir(out_dir)
    
    builder = (
        SparkSession.builder.appName(name)
        .config("PYSPARK_PYTHON", "python3")
        .config("PYSPARK_DRIVER_PYTHON", "python3")
    )
    
    if debug:
        builder = builder.master("local[2]")
        
    return(builder.getOrCreate())



if __name__ == "__main__":
    # initialize spark session
    
    spark = start_spark("gender_decahose", debug = False)
    
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    print("Reading Data")
    
    df = (
        spark.read.json(PANEL_TWEETS_LOC,
                        schema=StructType.fromJson(json.load(open(TWEETS_SCHEMA))))
        .selectExpr(
            "id_str",
            "created_at",
            "retweeted_status.id_str as retweet_id",
            "user.screen_name as handle",
            "user.name as display_name",
            "user.id_str as userid",
            "user.description as bio",
            "user.followers_count as followers",
            "user.friends_count as followees",
            "retweet_count as retweets",
            "favorite_count as likes"
        )
        .withColumn("parsed_date",
                    F.to_timestamp(F.col('created_at'), "EEE MMM dd HH:mm:ss ZZZZZ yyyy"))
        .orderBy("parsed_date")
    )
    
    print("Number of tweets, ", df.count())
    
    df = df.filter(F.isnan(df.retweet_id) | F.col("retweet_id").isNull() | (F.length(F.col("retweet_id")) == 0))
    
    print("Number of tweets without retweets ", df.count())
    
    print("calculating")
    
    out = df.groupBy('userid').agg(
        F.sum('retweets').alias('retweet_sum'),
        F.sum('likes').alias('likes_sum'),
        F.mean('retweets').alias('retweet_avg'),
        F.mean('likes').alias('likes_avg'),
        F.last('followers').alias('followers'),
        F.last('followees').alias('followees'),
        F.last('handle').alias('handle'),
        F.last('display_name').alias('display_name'),
        F.count("*").alias("n_tweets")
    )
    
    print("writing")
    
    out.write.csv(out_dir, header = True, sep = "\t")
    
    agg_cmd = "hdfs dfs -cat {}/* | uniq > /attention_panel.tsv".format(out_dir)
    subprocess.run(agg_cmd, shell=True)
    
    
    
    