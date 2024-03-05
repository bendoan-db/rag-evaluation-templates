# Databricks notebook source
# MAGIC %pip install langsmith evaluate langchain_core openai

# COMMAND ----------

# MAGIC %pip install langchain_openai --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langsmith.evaluation import EvaluationResult, run_evaluator, RunEvaluator
from langsmith.schemas import Example, Run
from typing import Optional
from evaluate import load

@run_evaluator
def is_empty(run: Run, example: Example | None = None):
    model_outputs = run.outputs["output"]
    score = not model_outputs.strip()
    return EvaluationResult(key="is_empty", score=score)

#You may want to parametrize your evaluator as a class. 
#For this, you can use the RunEvaluator class, which is functionally equivalent to the decorator above.
class BlocklistEvaluator(RunEvaluator):
    def __init__(self, blocklist: list[str]):
        self.blocklist = blocklist

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult:
        model_outputs = run.outputs["output"]
        score = not any([word in model_outputs for word in self.blocklist])
        return EvaluationResult(key="blocklist", score=score)

class PerplexityEvaluator(RunEvaluator):
    def __init__(self, prediction_key: Optional[str] = None, model_id: str = "gpt-2"):
        self.prediction_key = prediction_key
        self.model_id = model_id
        self.metric_fn = load("perplexity", module_type="metric")

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")
        prediction = run.outputs[self.prediction_key]
        results = self.metric_fn.compute(
            predictions=[prediction], model_id=self.model_id
        )
        ppl = results["perplexities"][0]
        return EvaluationResult(key="Perplexity", score=ppl)

# COMMAND ----------

import re
from typing import Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain import openai
from langchain.evaluation import StringEvaluator

class RelevanceEvaluator(StringEvaluator):
    """An LLM-based relevance evaluator."""

    def __init__(self):
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        template = """On a scale from 0 to 100, how relevant is the following response to the input:
        --------
        INPUT: {input}
        --------
        OUTPUT: {prediction}
        --------
        Reason step by step about why the score is appropriate, then print the score at the end. At the end, repeat that score alone on a new line."""

        self.eval_chain = PromptTemplate.from_template(template) | llm

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "scored_relevance"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        evaluator_result = self.eval_chain.invoke(
            {"input": input, "prediction": prediction}, kwargs
        )
        reasoning, score = evaluator_result["text"].split("\n", maxsplit=1)
        score = re.search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip()) / 100.0
        return {"score": score, "reasoning": reasoning.strip()}

# COMMAND ----------


