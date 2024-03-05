# Databricks notebook source
# MAGIC %run ./setup/load_credentials

# COMMAND ----------

# MAGIC %pip install beautifulsoup4 seaborn

# COMMAND ----------

import pandas as pd
import mlflow
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# COMMAND ----------

request_log = spark.table("headlamp.dev.example_evaluation_set_request_log")

# COMMAND ----------

from pyspark.sql.functions import explode
retrievals = request_log.select("request_id",,explode("trace.steps")).select("col.retrieval.chunks")

# COMMAND ----------


