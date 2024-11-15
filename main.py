import os
import re
import json
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_FOLDER = config['data_folder']
SENTIMENT_MODEL = config['sentiment_model']
JUDGE_MODEL = config['llm_judge_model']
SUBJECT_OF_INTEREST_PARTICIPANTS = config['subject_of_interest_participants']
OPENAI_API_KEY = config['openai_api_key']

openai.api_key = OPENAI_API_KEY

# plot styles
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (16, 4)
plt.rcParams['font.size'] = 16

# Data import
def load_transcripts(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                data.append({'Filename': filename, 'Transcript': text})
    df = pd.DataFrame(data)
    df['digit'] = df['Filename'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(by='digit').drop(columns='digit').reset_index(drop=True)
    return df

df = load_transcripts(DATA_FOLDER)

# Section 1: EDA

def identify_participants(df, text_column='Transcript', subject_of_interest_participants=None):
    unique_participants = set()
    excluded_participants = set()
    for content in df[text_column]:
        lines = content.split("\n")
        for line in lines:
            if ":" in line:
                participant = line.split(":")[0].strip()
                if subject_of_interest_participants:
                    if participant in subject_of_interest_participants:
                        unique_participants.add(participant)
                    else:
                        excluded_participants.add(participant)
                else:
                    unique_participants.add(participant)
    for participant in unique_participants:
        df[participant] = df[text_column].apply(lambda x: 1 if f"{participant}:" in x else 0)
    df['Number of Participants'] = df[list(unique_participants)].sum(axis=1)
    return df, unique_participants, excluded_participants

df, unique_participants, excluded_participants = identify_participants(
    df, subject_of_interest_participants=SUBJECT_OF_INTEREST_PARTICIPANTS
)

print('Subject of interest participants detected in data:', unique_participants)
print('Possible participants in data, for your information:', excluded_participants)
print('Number of conversations with more than two participants:', len(df[df['Number of Participants'] > 2]))

def add_text_metrics(df, text_column='Transcript'):
    df['Character Count'] = df[text_column].apply(len)
    df['Word Count'] = df[text_column].apply(lambda x: len(x.split()))
    df['Sentence Count'] = df[text_column].apply(lambda x: len(re.split(r'[.!?]', x)) - 1)
    return df

df = add_text_metrics(df)

# EDA Plots
def plot_eda(df, columns):
    os.makedirs('plots', exist_ok=True)
    for column in columns:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)

        plt.tight_layout()
        plt.savefig(f'plots/{column}_eda_plot.png')

eda_columns = ['Character Count', 'Word Count', 'Sentence Count']
plot_eda(df, eda_columns)

# Section 2: Sentiment Analysis Using LLM

def analyse_sentiment(transcript):
    prompt = f"""
You are an expert in customer service in the insurance industry. Please analyse the conversation (call transcript) given below for the sentiment of the customer (referred to as 'Member') according to the key metrics below. Provide both a score for each metric and a concise, evidence-based explanation to explain why the score is given. Ensure you are considering only the customer's (Member's) sentiment for all metrics and scores. Use British English throughout your response. Make sure not to consider any PII data when scoring or providing an explanation. Use the word 'Member' instead of the actual name of the customer or member.

Output Format:
Provide each metric with the following structure:
- Metric Name (e.g., Sentiment Score, Call Outcome, Sentiment Change)
- Score/Outcome (Use the specified scoring format for each metric below)
- Explanation (A brief, evidence-backed summary explaining why the score was suggested)

Metrics and Scoring Instructions:
1. Sentiment Score
   - Score: Assign a sentiment score based on the overall tone of the conversation. Use "Positive", "Neutral", or "Negative".
   - Explanation: Explain what score should be given with specific evidence from the conversation, noting any key phrases, tone of language, or mood conveyed by the customer that supports the sentiment rating.

2. Call Outcome (Resolution Status)
   - Outcome: Indicate the primary result of the conversation as "Issue Resolved" or "Follow-up Action Required".
   - Explanation: Provide reasoning based on whether the customer confirmed resolution or expressed unresolved concerns. Note any language suggesting satisfaction or the need for further assistance.

3. Sentiment Change
   - Score: Assess if the sentiment changed over the course of the conversation, using "Improved", "Declined", or "No Change".
   - Explanation: Describe the shift in sentiment from the beginning to the end, with evidence from the customer's initial and final messages to support the score. Do not consider if the issue has been resolved or not; instead, try to understand if there is improvement or decline in the sentiment of the customer.

Final Output format (for a completed analysis):
The final output should be a JSON-formatted dictionary without any new lines or \n, ensuring it is exactly a Python dictionary format. For example:

{{
    "sentiment_score": "your suggested score for sentiment_score",
    "sentiment_score_explanation": "explanation for sentiment_score",
    "call_outcome": "your suggested outcome for call_outcome",
    "call_outcome_explanation": "explanation for call_outcome",
    "sentiment_change": "your suggested score for sentiment_change",
    "sentiment_change_explanation": "explanation for sentiment_change"
}}

Conversation (call transcript): "{transcript}"
"""

    response = openai.ChatCompletion.create(
        model=SENTIMENT_MODEL,
        temperature=0.1,
        max_tokens=1000,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    result = response['choices'][0]['message']['content']

    analysis = json.loads(result)
    return analysis

analysis_results = []
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    print(f"Analyzing transcript {idx+1}/{df.shape[0]}...")
    transcript = row['Transcript']
    try:
        analysis = analyse_sentiment(transcript)
        analysis_results.append(analysis)
    except Exception as e:
        print(f"Error analyzing transcript {idx+1}: {e}")
        analysis_results.append({})

analysis_df = pd.DataFrame(analysis_results)
df = pd.concat([df, analysis_df], axis=1)
os.makedirs('result', exist_ok=True)
df.to_csv('result/df_sentiment_analysis.csv', index=False, encoding='utf-8')

# Section 3: LLM-as-a-Judge Evaluator

def evaluate_sentiment_analysis(transcript, metric_name, score, explanation):
    prompt = f"""
You are an expert in customer service in the insurance industry. Please evaluate the customer's sentiment analysis that has been done on the conversation (call transcript) given below. The goal is to assess how well the sentiment analysis of the customer (referred to as 'Member') has been performed.

Metrics and Scoring Instructions:
- Factuality Score ({metric_name}_explanation_factuality) (1 to 5): This score evaluates how much of the content in the explanation is directly sourced from the conversation. A score of 1 indicates that the explanation has minimal or no factual basis from the conversation, while a score of 5 means the explanation is highly factual, with all statements rooted directly in the conversation's content.
- Completeness Score ({metric_name}_explanation_completeness) (1 to 5): This score assesses whether all elements from the conversation relevant to sentiment analysis have been considered in the explanation. A score of 1 indicates that many points were missed, resulting in an incomplete analysis. A score of 5 means that all relevant points from the conversation have been accounted for, making the explanation thorough and comprehensive.
- Sentiment Consistency ({metric_name}_consistency) (0 or 1): This binary metric evaluates whether the sentiment analysis score aligns well with the explanation provided. A value of 1 indicates that the explanation's reasoning supports the given sentiment score, whereas a value of 0 means there's a misalignment between the explanation and the sentiment score.

Final Output format:
The final output should be a JSON-formatted dictionary without any new lines or \n, ensuring it is exactly a Python dictionary format. For example:

{{
    "{metric_name}_explanation_factuality": "your suggested score",
    "{metric_name}_explanation_completeness": "your suggested score",
    "{metric_name}_consistency": "your suggested value"
}}

Sentiment analysis that has been done:
{metric_name.capitalize().replace('_', ' ')}:
Score: "{score}"
Explanation: "{explanation}"

Conversation (call transcript): "{transcript}"
"""

    response = openai.ChatCompletion.create(
        model=JUDGE_MODEL,
        temperature=0.1,
        max_tokens=1000,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )

    result = response['choices'][0]['message']['content']

    evaluation = json.loads(result)
    return evaluation

evaluation_results = []
metrics = ['sentiment_score', 'call_outcome', 'sentiment_change']

for metric in metrics:
    temp_results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        print(f"Evaluating {metric} for transcript {idx+1}/{df.shape[0]}...")
        transcript = row['Transcript']
        score = row[metric]
        explanation = row[f"{metric}_explanation"]
        try:
            evaluation = evaluate_sentiment_analysis(transcript, metric, score, explanation)
            temp_results.append(evaluation)
        except Exception as e:
            print(f"Error evaluating {metric} for transcript {idx+1}: {e}")
            temp_results.append({})
    evaluation_df = pd.DataFrame(temp_results)
    df = pd.concat([df, evaluation_df], axis=1)

os.makedirs('result', exist_ok=True)
df.to_csv('result/df_final.csv', index=False, encoding='utf-8')