# Databricks notebook source
# MAGIC %pip install langsmith

# COMMAND ----------

# MAGIC %pip install langchain_core --upgrade langchain_openai --upgrade

# COMMAND ----------

# MAGIC %pip install langchain --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ["LANGCHAIN_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="langsmith-key")
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="openai-key")
os.environ["LANGCHAIN_PROJECT"] = "databricks-field-demo-test"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# COMMAND ----------

from langsmith import Client

# Inputs are provided to your model, so it know what to generate
dataset_inputs = [
  "a rap battle between Atticus Finch and Cicero",
  "a rap battle between Barbie and Oppenheimer",
  # ... add more as desired
]

# Outputs are provided to the evaluator, so it knows what to compare to
# Outputs are optional but recommended.
dataset_outputs = [
    {"must_mention": ["lawyer", "justice"]},
    {"must_mention": ["plastic", "nuclear"]},
]
client = Client()
dataset_name = "Rap Battle Dataset"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Rap battle prompts.",
)
client.create_examples(
    inputs=[{"question": q} for q in dataset_inputs],
    outputs=dataset_outputs,
    dataset_id=dataset.id,
)

# COMMAND ----------

import openai
# You evaluate any arbitrary function over the dataset.
# The input to the function will be the inputs dictionary for each example.
def predict_result(input_: dict) -> dict:
    messages = [{"role": "user", "content": input_["question"]}]
    response = openai.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
    return {"output": response}

# COMMAND ----------

from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith import Client
from langchain_openai import OpenAI

#create custom evaluator
@run_evaluator
def must_mention(run, example) -> EvaluationResult:
    prediction = run.outputs["generations"][0][0]["text"]
    required = example.outputs.get("must_mention") or []
    score = all(phrase in prediction for phrase in required)
    return EvaluationResult(key="must_mention", score=score)

llm = OpenAI()

eval_config = RunEvalConfig(
    custom_evaluators=[must_mention],
    # You can also use a prebuilt evaluator
    # by providing a name or RunEvalConfig.<configured evaluator>
    evaluators=[
        # You can specify an evaluator by name/enum.
        # In this case, the default criterion is "helpfulness"
        "criteria",
        # Or you can configure the evaluator
        RunEvalConfig.Criteria("harmfulness"),
        RunEvalConfig.Criteria(
            {
                "cliche": "Are the lyrics cliche?"
                " Respond Y if they are, N if they're entirely unique."
            }
        ),
    ],
)
client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=llm,
    evaluation=eval_config,
    verbose=True,
    project_name="chatopenai-test-1",
)

# COMMAND ----------


