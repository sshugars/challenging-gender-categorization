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

TWEETS_LOC = "hdfs://decahose_tweets/"
TWEETS_SCHEMA = "tweet_schema.json"

out_dir = "hdfs://out_dir/"

day_list = ['2021-' + x + '-11' for x in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']]

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
    spark = start_spark("gender_decahose", debug = DEBUG)

    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
    
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set("io.compression.codecs","io.sensesecure.hadoop.xz.XZCodec")
    sc.setLogLevel("OFF")

    print("Reading Data")
    
    t1 = time.time()
    
    df = (spark.read.format("json").load(TWEETS_LOC + "tweets.json." + day_list[0] + '.xz')
        .selectExpr(
            "lang",
            "user.id_str as userid",
            "user.description as bio",
            "user.screen_name as handle",
            "user.name as display_name"
        )
        .filter(F.col("lang") == "en")
        .drop_duplicates(["userid"])
    )
    
    print("Read ", day_list[0], " ", (time.time()-t1)/3600)
    
    for day in day_list[1:len(day_list)]:
        df_new = (
            spark.read.format("json").load(TWEETS_LOC + "tweets.json." + day + '.xz')
            .selectExpr(
                "lang",
                "user.id_str as userid",
                "user.description as bio",
                "user.screen_name as handle",
                "user.name as display_name"
            )
            .filter(F.col("lang") == "en")
            .drop_duplicates(["userid"])
        )
        df = df.union(df_new)
        print("Read", day, " ", (time.time()-t1)/3600)


        
    print("Number of tweets", df.count())
    
    df = (
        df.drop_duplicates(["userid"])
        .withColumn("bio", F.regexp_replace("bio", r'[\t\n\r\f\v ]+' , ' '))
        .withColumn("display_name", F.regexp_replace("display_name", r'[\t\n\r\f\v ]+' , ' '))
        .withColumn("display_name", F.regexp_replace("display_name", r'/\s+/g' , ' '))
        .select("userid", "bio", "display_name", "handle")
    )
    
    print("Total number of bios", df.count())
    df.write.csv(out_dir, header = True, sep = "\t")
    
    agg_cmd = "hdfs dfs -cat {}/* | uniq > /bios_deca.tsv".format(out_dir)
    subprocess.run(agg_cmd, shell=True)
