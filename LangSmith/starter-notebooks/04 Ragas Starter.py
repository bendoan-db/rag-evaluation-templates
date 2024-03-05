# Databricks notebook source
# MAGIC %md 
# MAGIC source: https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
# MAGIC
# MAGIC **Note**: Ragas support is currently not available for LangChain 0.1+ and Langchain <0.1 creates depedency issues with Langsmith, so this notebook does not currently work. Ragas support for langchain & langsmith are coming out in langchain 0.2.
# MAGIC
# MAGIC Follow git issue here
# MAGIC - https://github.com/explodinggradients/ragas/issues/567
# MAGIC - https://github.com/explodinggradients/ragas/issues/571

# COMMAND ----------

# MAGIC %run ../setup/init

# COMMAND ----------

# MAGIC %pip install ragas==0.0.22 chromadb

# COMMAND ----------

# MAGIC %pip install -U sqlalchemy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../setup/load_credentials

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# load the Wikipedia page and create index
loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City")
index = VectorstoreIndexCreator().from_loaders([loader])

# create the QA chain
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=index.vectorstore.as_retriever(), return_source_documents=True
)

# testing it out
question = "How did New York City get its name?"
result = qa_chain({"query": question})
result["result"]

# output
# 'New York City got its name in 1664 when it was renamed after the Duke of York, who later became King James II of England. The city was originally called New Amsterdam by Dutch colonists and was renamed New York when it came under British control.'

# COMMAND ----------

from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain.evalchain import RagasEvaluatorChain

# make eval chains
eval_chains = {
    m.name: RagasEvaluatorChain(metric=m) 
    for m in [faithfulness, answer_relevancy, context_relevancy]
}

# COMMAND ----------

eval_chains

# COMMAND ----------

for name, eval_chain in eval_chains.items():
    score_name = f"{name}_score"
    print(f"{score_name}: {eval_chain(result)[score_name]}")

# COMMAND ----------

# test dataset
eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
]

eval_answers = [
    "8,804,000", # incorrect answer
    "Queens",    # incorrect answer
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, and educational institutions further fuel its economic prowess. The city's transportation network and global influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the lands to his brother, the Duke of York, who named the city New York in his own honor.",
    'The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.',
]

examples = [{"query": q, "ground_truths": [eval_answers[i]]} 
           for i, q in enumerate(eval_questions)]

# COMMAND ----------

# dataset creation
from langsmith import Client
from langsmith.utils import LangSmithError

client = Client()
dataset_name = "NYC test"

try:
    # check if dataset exists
    dataset = client.read_dataset(dataset_name=dataset_name)
    print("using existing dataset: ", dataset.name)
except LangSmithError:
    # if not create a new one with the generated query examples
    dataset = client.create_dataset(
        dataset_name=dataset_name, description="NYC test dataset"
    )
    for q in eval_questions:
        client.create_example(
            inputs={"query": q},
            dataset_id=dataset.id,
        )

    print("Created a new dataset: ", dataset.name)

# COMMAND ----------

# factory function that return a new qa chain
def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=return_context,
    )
    return qa_chain

# COMMAND ----------

from langchain.smith import RunEvalConfig, run_on_dataset

evaluation_config = RunEvalConfig(
    custom_evaluators=[eval_chains.values()],
    prediction_key="result",
)
    
result = run_on_dataset(
    client,
    dataset_name,
    create_qa_chain,
    evaluation=evaluation_config,
    input_mapper=lambda x: x,
)

# COMMAND ----------


