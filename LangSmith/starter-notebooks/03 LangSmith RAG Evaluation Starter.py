# Databricks notebook source
# MAGIC %run ../setup/init

# COMMAND ----------

# MAGIC %pip install langchain --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../setup/load_credentials

# COMMAND ----------

import langchain
print(langchain.__version__)

# COMMAND ----------

# A simple example dataset
examples = [
    {
        "inputs": {
            "question": "What's the company's total revenue for q2 of 2022?",
            "documents": [
                {
                    "metadata": {},
                    "page_content": "In q1 the lemonade company made $4.95. In q2 revenue increased by a sizeable amount to just over $2T dollars."
                }
            ],
        },
        "outputs": {
            "label": "2 trillion dollars",
        },
    },
    {
        "inputs": {
            "question": "Who is Lebron?",
            "documents": [
                {
                    "metadata": {},
                    "page_content": "On Thursday, February 16, Lebron James was nominated as President of the United States."
                }
            ],
        },
        "outputs": {
            "label": "Lebron James is the President of the USA.",
        },
    }
]

# COMMAND ----------

import uuid
uid = uuid.uuid4()

# COMMAND ----------

import os
from langsmith import Client

os.environ["LANGCHAIN_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="langsmith-key")
client = Client()

dataset_name = "Faithfulness Example - {uuid}"
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[e["inputs"] for e in examples], 
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

# COMMAND ----------

from langchain import chat_models, prompts
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.retriever import BaseRetriever
from langchain_core import documents
from langchain_openai import ChatOpenAI

class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Example")]

# This is what we will evaluate
response_synthesizer = (
    prompts.ChatPromptTemplate.from_messages(
        [
            ("system", "Respond using the following documents as context:\n{documents}"),
            ("user", "{question}")
        ]
    ) | ChatOpenAI()
)

# Full chain below for illustration
chain = (
    {
        "documents": MyRetriever(),
        "qusetion": RunnablePassthrough(),
    }
    | response_synthesizer
)

# COMMAND ----------

from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator

class FaithfulnessEvaluator(RunEvaluator):
    """
    Define a custom "FaithfulnessEvaluator" that measures how faithful the chain's output prediction is to the reference input documents, given the user's input question.
    """
    def __init__(self):
        self.evaluator = load_evaluator(
            "labeled_score_string", 
            criteria={"faithful": "How faithful is the submission to the reference context?"},
            normalize_by=10,
        )

    def evaluate_run(self, run, example) -> EvaluationResult:
        res = self.evaluator.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["question"],
            # We are treating the documents as the reference context in this case.
            reference=example.inputs["documents"],
        )
        return EvaluationResult(key="labeled_criteria:faithful", **res)

# COMMAND ----------

from langchain.smith import RunEvalConfig

eval_config = RunEvalConfig(
    evaluators=["qa"],
    custom_evaluators=[FaithfulnessEvaluator()],
    input_key="question",
)
results = client.run_on_dataset(
    llm_or_chain_factory=response_synthesizer,
    dataset_name=dataset_name,
    evaluation=eval_config,
)   

# COMMAND ----------


