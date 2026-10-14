# 🤖 GenAI Foundations & LLM Exploration Toolkit

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-1C3C3C.svg?style=flat&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000.svg?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-SDK-412991.svg?style=flat&logo=openai&logoColor=white)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97706.svg?style=flat&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Package Manager: uv](https://img.shields.io/badge/managed_by-uv-DE5FE9.svg?style=flat)](https://github.com/astral-sh/uv)

A hands-on, practical toolkit exploring the core foundations of modern **Generative AI**, **Large Language Models (LLMs)**, and **Natural Language Processing (NLP)**.

This repository demonstrates real-world implementations across both local open-source models (via **Ollama** and **Hugging Face**) and leading proprietary APIs (**OpenAI** and **Anthropic Claude**), tied together with **LangChain (LCEL)** and managed with modern Python tooling (**uv**).

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Modules & Core Features](#-modules--core-features)
  - [1. Interactive CLI Chatbot (`main.py`)](#1-interactive-cli-chatbot-mainpy)
  - [2. Hugging Face NLP Pipeline (`notebook.ipynb`)](#2-hugging-face-nlp-pipeline-notebookipynb)
  - [3. OpenAI API Integration (`notebook2.ipynb`)](#3-openai-api-integration-notebook2ipynb)
  - [4. Anthropic Claude & Temperature Tuning (`notebook3.ipynb`)](#4-anthropic-claude--temperature-tuning-notebook3ipynb)
- [Prerequisites](#-prerequisites)
- [Quickstart & Installation](#-quickstart--installation)
- [Environment Configuration](#-environment-configuration)
- [Usage Instructions](#-usage-instructions)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🚀 Overview

This project serves as an applied lab for mastering GenAI workflows:

- **Local Inference with LangChain Expression Language (LCEL)**: Build robust conversational pipelines with dynamic prompt templates, system instructions, and stateful multi-turn history.
- **NLP Text Classification**: Run pretrained transformer models locally using Hugging Face's pipeline abstractions.
- **Provider-Agnostic LLM Access**: Connect to cloud providers or custom gateway proxies with flexible base URLs and API keys.
- **Temperature & Sampling Exploration**: Observe the impact of model decoding parameters (e.g., low temperature `0.1` for factual responses vs. high temperature `1.5` for creative diversity).
- **Modern Python Environment**: Fast, deterministic package resolution and virtual environment management via `uv`.

---

## 📁 Repository Structure

```text
.
├── main.py              # Stateful interactive CLI assistant using LangChain & Ollama (Gemma 3)
├── notebook.ipynb       # Hugging Face Transformers pipeline (text classification with DistilBERT)
├── notebook2.ipynb      # OpenAI SDK client integration with customizable endpoints and model selection
├── notebook3.ipynb      # Anthropic Claude SDK with prompt experimentation & temperature tuning
├── .env.example         # Template for environment variables (API keys, Base URLs, model names)
├── .gitignore           # Ignores virtual environments, cache, and sensitive .env files
├── pyproject.toml       # Project metadata, dependencies, and Python constraints
├── uv.lock              # Pinned, reproducible dependency lockfile
└── README.md            # Comprehensive project documentation
```

---

## 🧩 Modules & Core Features

### 1. Interactive CLI Chatbot (`main.py`)
- **Engine**: LangChain with `ChatOllama` running local `gemma3:4b` (`temperature = 0.7`).
- **Architecture**:
  - `ChatPromptTemplate` with structured system role: `"You are a helpful assistant."`
  - `MessagesPlaceholder(variable_name="chat_history")` preserves conversational context across turns.
  - LCEL pipeline: `chain = prompt | llm | StrOutputParser()`.
  - Continuous conversation loop in the terminal; exit anytime by typing `exit`.

### 2. Hugging Face NLP Pipeline (`notebook.ipynb`)
- **Model**: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
- **Functionality**: Uses Hugging Face's `pipeline("text-classification")` to perform fast, on-device sentiment and sequence classification on input text.

### 3. OpenAI API Integration (`notebook2.ipynb`)
- **SDK**: `openai>=3.24.0`.
- **Flexibility**: Configured via environment variables (`BASE_URL`, `API_KEY`, `MODEL_NAME`).
- **Capability**: Enables chat completion queries against standard OpenAI endpoints or custom proxy gateways.

### 4. Anthropic Claude & Temperature Tuning (`notebook3.ipynb`)
- **SDK**: `anthropic>=1.11.0`.
- **Experimentation**: Evaluates creative naming tasks under different sampling temperatures:
  - `temperature = 0.1`: Highly focused, deterministic outputs.
  - `temperature = 1.5`: Diverse, exploratory, and creative generation.

---

## ⚙️ Prerequisites

1. **Python 3.13+** installed on your system.
2. **uv** (recommended for ultra-fast dependency management):
   - **Windows (PowerShell)**:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **macOS / Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
3. **Ollama** installed and running locally:
   - Download from [ollama.com](https://ollama.com/).
   - Pull the default local model:
     ```bash
     ollama pull gemma3:4b
     ```

---

## 🛠️ Quickstart & Installation

### 1. Clone or Open the Workspace
```bash
cd "c:/Abdullah Ahmed/Ai course/Project/new"
```

### 2. Install Dependencies

Using **`uv`** (recommended):
```bash
uv sync
```

Alternatively, using standard **`pip`**:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e .
```

---

## 🔐 Environment Configuration

Create a `.env` file from the provided `.env.example` template:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` and fill in your configuration:

```env
# Base URL for API requests (OpenAI endpoint, Anthropic endpoint, or custom proxy)
BASE_URL=https://api.openai.com/v1

# API Key for cloud providers
API_KEY=your_api_key_here

# Target Model Identifier (e.g. gpt-4o-mini, claude-3-5-sonnet-20241022, gemma3:4b)
MODEL_NAME=gpt-4o-mini
```

---

## 💻 Usage Instructions

### Running the Terminal Assistant (`main.py`)

Ensure your Ollama service is active, then execute:

```bash
uv run python main.py
```

*Example interaction:*
```text
Welcome to the AI assistant. Type 'exit' to quit.
Please enter your question: Who was Alan Turing?
Bot: Alan Turing was an English mathematician, computer scientist, and cryptanalyst...
Please enter your question: What was his main contribution?
Bot: Turing is widely considered the father of theoretical computer science and artificial intelligence...
Please enter your question: exit
Exiting the AI assistant. Goodbye!
```

### Running the Jupyter Notebooks

Launch Jupyter to explore the notebook experiments:

```bash
uv run jupyter lab
```

Or open the notebooks directly in **VS Code** and select the `.venv` Python kernel:
- `notebook.ipynb` — Hugging Face pipeline inference.
- `notebook2.ipynb` — OpenAI chat completions.
- `notebook3.ipynb` — Anthropic Claude messages and temperature experiments.

---

## 📦 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.13** | Core runtime environment |
| **LangChain (`langchain`, `langchain-ollama`)** | Orchestration, LCEL prompt chaining, and memory history |
| **Ollama (`ollama`)** | Local open-source model execution (`gemma3:4b`) |
| **Transformers & PyTorch (`torch`, `transformers`)** | Hugging Face deep learning models & NLP pipelines |
| **OpenAI SDK (`openai`)** | Client for OpenAI-compatible chat completion APIs |
| **Anthropic SDK (`anthropic`)** | Client for Claude Messages API with token & parameter controls |
| **python-dotenv** | Secure environment variable handling |
| **uv** | Modern, fast package resolution and virtual environment management |

---

## 👤 Author

- **Abdullah Ahmed**
  - Email: [abd1962964@gmail.com](mailto:abd1962964@gmail.com)
