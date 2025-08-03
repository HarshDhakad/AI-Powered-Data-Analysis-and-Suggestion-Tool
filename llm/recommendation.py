
import pandas as pd
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# ✅ Load API Token from Streamlit Secrets
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ✅ Initialize LLM with token
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# ✅ Wrap in Chat Model
model = ChatHuggingFace(llm=llm)

# Continue your prompt templates and chain setup...
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

parser = StrOutputParser()

def generate_dataset_summary(df, target_column):
    rows, cols = df.shape
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].to_dict()

    missing_info = "\nNo missing values detected."
    if missing_summary:
        missing_info = "\nColumns with missing values:\n"
        for col, count in missing_summary.items():
            missing_info += f"- {col}: {count} missing\n"

    summary = f"""
Dataset contains {rows} rows and {cols} columns.

Target Column: {target_column}

Feature Columns:
- {', '.join(feature_columns)}

Column Types:
- Numeric Columns: {', '.join(numeric_cols)}
- Categorical Columns: {', '.join(categorical_cols)}

{missing_info}

This dataset aims to predict '{target_column}'.
"""
    return summary.strip()


def generate_predictions_summary(model_type, predictions):
    if not isinstance(predictions, list):
        predictions = pd.Series(predictions)

    if model_type == 'classification':
        class_distribution = predictions.value_counts(normalize=True) * 100
        summary = "Model predicted class distribution:\n"
        for label, percent in class_distribution.items():
            summary += f"- Class '{label}': {percent:.2f}%\n"
        summary += f"\nDominant Class: {class_distribution.idxmax()} ({class_distribution.max():.2f}%)."

    elif model_type == 'regression':
        summary = (
            f"Model predicted numeric values:\n"
            f"- Average: {predictions.mean():.2f}\n"
            f"- Max: {predictions.max():.2f}\n"
            f"- Min: {predictions.min():.2f}"
        )
    else:
        summary = "Unknown model type."

    return summary

prompt_template = PromptTemplate(
    template="""You are an expert data analyst.

Dataset Summary:
{dataset_summary}

Predictions Summary:
{predictions_summary}

Problem Statement:
{problem_statement}

Based on the dataset summary and the prediction summary, generate 2-3 actionable business recommendations to help resolve the problem described above.""",
    input_variables=["dataset_summary", "predictions_summary", "problem_statement"]
)

chain = prompt_template | model | parser

def generate_llm_recommendation(problem_statement, dataset_summary, predictions_summary):
    return chain.invoke({
        "dataset_summary": dataset_summary,
        "predictions_summary": predictions_summary,
        "problem_statement": problem_statement
    })
