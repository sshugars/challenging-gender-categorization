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




PANEL_TWEETS_LOC = "hdfs://tweets/"
TWEETS_SCHEMA = "tweet_schema.json"

out_dir = "hdfs://out_dir/"


if __name__ == "__main__":
    # initialize spark session
    delete_spark_staging_dir(out_dir)

    spark = (
        SparkSession.builder.appName("twitter_gender")
        .config("PYSPARK_PYTHON", "python3")
        .config("PYSPARK_DRIVER_PYTHON", "python3")
        .getOrCreate()
    )
    
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    print("Reading Data")
    
    read = (
        spark.read.json(PANEL_TWEETS_LOC,
                        schema=StructType.fromJson(json.load(open(TWEETS_SCHEMA))))
        .selectExpr(
            "id_str",
            "created_at",
            "user.screen_name as handle",
            "user.name as display_name",
            "user.id_str as userid",
            "user.description as bio")
        .withColumn("parsed_date",
                    F.to_timestamp(F.col('created_at'), "EEE MMM dd HH:mm:ss ZZZZZ yyyy"))
        .withColumn("Month",
                    F.month('parsed_date'))
    )
    
    n_tweets = read.groupBy("userid").count()
    
    read = read.drop_duplicates(
        ["userid"]
    ).join(
        n_tweets, "userid", "left"
    )

    
    print("Reading voter file data and matching")
        # Join matches with the voter file data
    panel_data = read_panel_data(spark, "hdfs://voter_file.csv")

    read = read.join(
        panel_data, read.userid == panel_data.twProfileID, "left"
    ).drop("twProfileID")

    # Output data
    read = read.select(
        "userid",
        "handle",
        "display_name",
        "bio",
        "state",
        "county",
        "race",
        "gender",
        "age",
        "party_score",
        "count"
    ).withColumnRenamed(
        "gender", "voter_file_sex"
    ).withColumnRenamed(
        "count", "n_tweets"
    )
       
    
    print("Total number of bios ", read.count())
    read.write.csv(out_dir, header = True, sep = "\t")
    
    agg_cmd = "hdfs dfs -cat {}/* | uniq > /bios_2021.tsv".format(out_dir)
    subprocess.run(agg_cmd, shell=True)