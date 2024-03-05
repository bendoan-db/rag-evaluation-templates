# Databricks notebook source
# MAGIC %md
# MAGIC ## Sources
# MAGIC - [LangSmith Evaluator Implementations](https://docs.smith.langchain.com/evaluation/faq/evaluator-implementations)
# MAGIC - [LangChain Cookbook on RAG Evaluations](https://docs.smith.langchain.com/cookbook/testing-examples/using-fixed-sources)

# COMMAND ----------

# MAGIC %run ./setup/init

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 transformers==4.30.2 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./setup/load_credentials

# COMMAND ----------

# DBTITLE 1,Creds for VS Index
#this assumes you've created a search index with the databricks documentation, per the rag-db-demo
import os

#set langchain project as env variable. Token is set in load_credentials
os.environ["LANGCHAIN_PROJECT"] = "databricks-field-demo-test"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

catalog="doan_demo_catalog"
schema="rag_chatbot"
VECTOR_SEARCH_ENDPOINT_NAME = "dbdemos_vs_endpoint"
vs_index_fullname = f"{catalog}.{schema}.databricks_documentation_vs_index"

# COMMAND ----------

from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks
from langsmith import traceable

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
def get_retriever(persist_dir: str = None, endpoint_name:str=None, index_name:str=None, embedding_model=None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

from langchain.chat_models import ChatDatabricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


@traceable(name="Chat Pipeline Trace")
def chat_pipeline(question, chat_model, embedding_model, endpoint_name, index_name): 

  TEMPLATE = """You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
  Use the following pieces of context to answer the question at the end:
  {context}
  Question: {question}
  Answer:
  """
  
  prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])
  chain = RetrievalQA.from_chain_type(
      llm=chat_model,
      chain_type="stuff",
      retriever=get_retriever(endpoint_name=endpoint_name, index_name=index_name, embedding_model=embedding_model),
      chain_type_kwargs={"prompt": prompt}
  )
  answer = chain.run(question)
  return answer

# COMMAND ----------

question={"query": "How can I speed up the MERGE operations on my Delta tables?"}
chat_pipeline(question, chat_model=chat_model, embedding_model=embedding_model, endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=vs_index_fullname)

# COMMAND ----------

question={"query": "What are some limitations when using Delta Live Tables and Unity Catalog together?"}
chat_pipeline(question, chat_model=chat_model, embedding_model=embedding_model, endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=vs_index_fullname)

# COMMAND ----------


