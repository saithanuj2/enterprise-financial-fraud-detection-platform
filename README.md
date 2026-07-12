# 🛡️ Enterprise Financial Fraud Detection Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![dbt](https://img.shields.io/badge/dbt-Core-FF694B?style=for-the-badge&logo=dbt)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0099CC?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-black?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

</p>

---

# 📌 Project Overview

The **Enterprise Financial Fraud Detection Platform** is an end-to-end AI-powered fraud analytics solution designed to detect suspicious financial transactions, generate machine learning predictions, and assist fraud investigators using Retrieval-Augmented Generation (RAG).

The platform combines modern data engineering, machine learning, vector search, and Large Language Models (LLMs) to simulate how enterprise financial institutions investigate fraudulent activities.

This project demonstrates an enterprise architecture using **PostgreSQL**, **dbt**, **Machine Learning**, **FAISS**, **Ollama (Llama 3.2)**, and **Streamlit**.

---

# 🚀 Key Features

✅ Enterprise PostgreSQL Data Warehouse

✅ Dockerized Infrastructure

✅ dbt Data Transformations

✅ Feature Engineering Pipeline

✅ Random Forest Fraud Detection Model

✅ Fraud Prediction Storage in PostgreSQL

✅ Retrieval-Augmented Generation (RAG)

✅ FAISS Vector Search

✅ Ollama + Llama 3.2 AI Investigator

✅ Enterprise Streamlit Dashboard

✅ Transaction Search & Investigation

✅ Model Performance Monitoring

---

# 💼 Business Problem

Financial institutions process millions of transactions every day.

Traditional fraud investigations rely heavily on manual review, making it difficult to quickly identify suspicious behavior.

Fraud analysts require intelligent systems capable of:

- Detecting suspicious transactions
- Prioritizing high-risk cases
- Retrieving similar historical fraud incidents
- Generating investigation reports
- Supporting data-driven fraud decisions

This platform addresses those challenges through a production-inspired AI workflow.

---

# 🏗️ Solution Architecture

> **Replace this section with your architecture diagram later**

```
                    PaySim Dataset
                          │
                          ▼
                  Python ETL Pipeline
                          │
                          ▼
             PostgreSQL Data Warehouse
                     (Docker)
                          │
                          ▼
                  dbt Transformations
                          │
                          ▼
              Fraud Feature Engineering
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Random Forest ML          Fraud Case Generator
            │                           │
            ▼                           ▼
 Fraud Predictions Table      Investigation Summaries
            │                           │
            └─────────────┬─────────────┘
                          ▼
                 FAISS Vector Database
                          │
                          ▼
                 Ollama (Llama 3.2)
                          │
                          ▼
            AI Fraud Investigation Engine
                          │
                          ▼
          Enterprise Streamlit Dashboard
```

---

# ⚙️ Technology Stack

| Layer | Technologies |
|---------|-------------|
| Programming | Python |
| Database | PostgreSQL |
| Data Warehouse | PostgreSQL |
| Data Modeling | dbt |
| Machine Learning | Scikit-Learn |
| Vector Database | FAISS |
| LLM | Ollama + Llama 3.2 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Containerization | Docker |
| Version Control | Git & GitHub |

---

# 🔄 End-to-End Workflow

```
PaySim Dataset
        │
        ▼
Python ETL Pipeline
        │
        ▼
PostgreSQL
        │
        ▼
dbt Data Models
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest Model
        │
        ▼
Fraud Predictions
        │
        ▼
Fraud Case Summaries
        │
        ▼
FAISS Vector Store
        │
        ▼
Llama 3.2 via Ollama
        │
        ▼
Enterprise Dashboard
```

---

# 🤖 Machine Learning Pipeline

### Dataset

- PaySim Financial Dataset

### Total Records

**6,362,620 Transactions**

### Model

Random Forest Classifier

### Performance

| Metric | Score |
|----------|---------|
| ROC AUC | **0.99966** |
| Precision | **0.98** |
| Recall | **1.00** |
| F1 Score | **0.99** |

Predictions are automatically written back into PostgreSQL for downstream analytics and AI-assisted investigations.

---

# 🧠 AI Fraud Investigation Assistant

The platform includes an enterprise Retrieval-Augmented Generation (RAG) assistant.

Workflow:

1. User asks an investigation question.
2. FAISS retrieves the most relevant historical fraud cases.
3. Ollama (Llama 3.2) analyzes the retrieved evidence.
4. The assistant generates:

- Investigation Summary
- Risk Indicators
- Similar Historical Cases
- Recommended Actions
- Supporting Evidence

Example:

> "Why are transactions that empty the origin account considered suspicious?"

The assistant retrieves historical fraud cases and generates a contextual investigation report for fraud analysts.

---

# 📊 Dashboard Features

The Streamlit dashboard provides:

- Executive KPI Dashboard
- Fraud Analytics
- Transaction Explorer
- AI Investigation Assistant
- Machine Learning Monitoring
- Fraud Prediction Summary
- PostgreSQL Health Monitoring
- Vector Database Status
- LLM Connectivity Checks

---

# 📸 Dashboard Preview

> Replace these placeholders with screenshots

```
screenshots/
│
├── dashboard.png
├── analytics.png
├── ai_assistant.png
├── monitoring.png
└── transaction_lookup.png
```

---

# 📂 Project Structure

```
enterprise-financial-fraud-detection-platform
│
├── config/
├── data/
├── src/
├── dbt_fraud_analytics/
├── rag_assistant/
│   ├── data/
│   ├── src/
│   └── vector_store/
│
├── tableau/
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/saithanuj2/enterprise-financial-fraud-detection-platform.git

cd enterprise-financial-fraud-detection-platform
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install packages

```bash
pip install -r requirements.txt
```

Start PostgreSQL

```bash
docker compose up -d
```

---

# ▶️ Run the Platform

Load Dataset

```bash
python src/load_to_postgres.py
```

Run dbt

```bash
cd dbt_fraud_analytics

dbt run

dbt test
```

Train Model

```bash
python src/train_model.py
```

Generate Fraud Cases

```bash
python rag_assistant/src/generate_case_summaries.py
```

Build Vector Store

```bash
python rag_assistant/src/build_vector_store.py
```

Launch Dashboard

```bash
streamlit run rag_assistant/src/app.py
```

---

# 📈 Project Highlights

✔ 6.3M+ Financial Transactions Processed

✔ PostgreSQL Data Warehouse

✔ dbt Feature Engineering

✔ Random Forest Fraud Detection

✔ 99.966% ROC AUC

✔ AI Investigation Assistant

✔ FAISS Semantic Search

✔ Ollama Llama 3.2 Integration

✔ Enterprise Streamlit Dashboard

✔ Docker Deployment

---

# 🔮 Future Enhancements

- Real-Time Kafka Streaming
- FastAPI REST APIs
- MLflow Model Registry
- SHAP Explainable AI
- Model Drift Detection
- Azure Deployment
- AWS Deployment
- Kubernetes Deployment
- CI/CD using GitHub Actions
- Role-Based Authentication

---

# 👨‍💻 Author

*# 👥 Project Team

- Sai Thanooj Kumar Revuru
- Shruti Lekkala
- Raja Shekar Reddy Akula

---
