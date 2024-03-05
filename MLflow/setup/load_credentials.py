# Databricks notebook source
import os
os.environ["LANGCHAIN_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="langsmith-key")
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="doan-demos", key="openai-key")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("doan-demos", "pat-token")
