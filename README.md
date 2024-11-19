# LangChain and Generative AI Experiments

Welcome to the LangChain and Generative AI Experiments repository! This repository contains various scripts and experiments demonstrating my studies on applications of the LangChain framework.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)

## Overview

This repository showcases multiple examples of how to use LangChain with different LLMs (Large Language Models), caching mechanisms, agents, tools, and integrations with external APIs. Some of the key topics covered include:

- Interacting with OpenAI API
- Caching LLM responses with InMemoryCache and SQLiteCache
- Streaming LLM responses
- Creating prompt templates
- Building and using simple and sequential chains
- Using LangChain Agents with Python REPL and other tools
- Integrating with DuckDuckGo and Wikipedia APIs
- Using Pinecone for vector stores
- Working with Google Generative AI (Gemini)
- Implementing a custom ChatGPT application
- Working with Memory in Chatbots (with and without limitation)
- Named Entity Extraction with structured output (pydantic model)
- Sentence classification applied to Incident report classification
- QA agent to a SQL database
- PDF document QA agent
- Basic RAG App for QA building and answer checking based on user input

## Installation

To run the scripts in this repository, you need to have Python 3.7 or higher installed. Follow the steps below to set up the environment:

1. Clone the repository:

2. Create a virtual environment:
    ```bash
    conda create -n <your_env_name> python=3.11
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the `env` directory and add your API keys and other necessary environment variables. For example:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```
