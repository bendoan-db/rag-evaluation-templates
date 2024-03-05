# Databricks notebook source
# MAGIC %pip install langsmith

# COMMAND ----------

# MAGIC %pip install langchain_openai --upgrade

# COMMAND ----------

# MAGIC %pip install langchain_core --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ["LANGCHAIN_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="langsmith-key")
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="openai-key")
os.environ["LANGCHAIN_PROJECT"] = "databricks-field-demo-test"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# COMMAND ----------

# To run the example below, ensure the environment variable OPENAI_API_KEY is set
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled

tracer = LangChainTracer(project_name="My Project")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you summarize this morning's meetings?"
context = "During this morning's meeting, we discussed LLM evaluations with much fervor. Two team members got into a fist fight over the correct tool to use for retrieval evaluations"
chain.invoke({"question": question, "context": context}, config={"callbacks": [tracer]})

# COMMAND ----------

# MAGIC %md Link to https://smith.langchain.com/o/541a2428-1385-4e4e-9b37-6ac3f4e6bf49/projects/p/5b93ba36-0770-4cd0-82bf-555c210e4c1e?timeModel=%7B%22duration%22%3A%227d%22%7D

# COMMAND ----------


