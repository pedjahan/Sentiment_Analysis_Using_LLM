# Sentiment Analysis with LLM with Evaluation using LLM-as-a-Judge

This project performs sentiment analysis on call transcripts using LLMs and evaluates the analysis using LLM-as-a-Judge. The project does not consider any PII masking, evaluation alignment with business (or human assessment/judgement) on the analysis provided by LLM-as-a-Judge and cannot handle large documents as it does not contain text chunking.

## Table of Contents

- [Introduction]
- [Project Structure]
- [Setup Instructions]
- [Usage]
- [License]

## Introduction

The goal of this project is to analyse customer (aka 'Member') sentiment from call transcripts using LLMs and then evaluate the quality of that analysis using an LLM-as-a-Judge. The project is divided into three main sections:

1. **Exploratory Data Analysis (EDA)**
2. **Sentiment Analysis using LLM**
3. **Evaluation of Sentiment Analysis using LLM-as-a-Judge**

## Project Structure

- `main.py`: The main Python script containing all the code.
- `config.yaml`: Configuration file containing settings such as folder name that contain text files of call transcript, LLM model names, API keys to connect to OpenAI API, etc.
- `requirements.txt`: A list of required Python libraries.
- `README.md`: This file.
- `data/`: A folder containing the transcript text filess.

## Setup Instructions

1. **Clone the Repository**
The project is unavailable online.

2. **Install Dependencies**
It’s recommended to use a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

3. **Configure settings**

Open config.yaml and update the following:
data_folder: Name of the folder containing text files (default is transcripts_v3).
sentiment_model: Model name for sentiment analysis (default is gpt-4o-mini).
llm_judge_model: Model name for LLM-as-a-Judge (default is gpt-4-turbo).
subject_of_interest_participants: List of participant names to consider (default is ['Member', 'Agent', 'Claims Review Team', 'Customer Support', 'PA Agent', 'Technical Support’])
openai_api_key: Your OpenAI API key which can be created in the following link: https://platform.openai.com/settings/organization/api-keys

5. **Run the Script file**
python main.py

This will perform EDA, sentiment analysis, and evaluation, and save the results to CSV file.

## License

TBC