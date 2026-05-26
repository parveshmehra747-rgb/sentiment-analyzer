# 🧠 Social Media Sentiment Analyzer

A real-time sentiment analysis web application for social media posts, built with Streamlit and powered by a RoBERTa model trained on 58 million tweets.

---

## 🔍 What It Does

This app analyzes the emotional tone of social media posts and classifies them as:

- 😊 **Positive** — happy, excited, or appreciative content
- 😠 **Negative** — angry, frustrated, or critical content  
- 😐 **Neutral** — factual, balanced, or emotionless content

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 Single Post Analysis | Paste any tweet or post and get instant sentiment + confidence score |
| 📊 Bulk Analysis | Analyze dozens of posts at once with charts and statistics |
| 🎯 Confidence Gauge | Visual gauge showing how confident the model is in its prediction |
| 📈 Charts & Graphs | Pie chart and bar chart showing sentiment distribution |
| ⬇️ CSV Export | Download all results as a spreadsheet with one click |
| ⚡ Example Posts | Built-in example posts to demo the app instantly |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Architecture | RoBERTa (Robustly Optimized BERT) |
| Training Data | 58 million tweets |
| Source | HuggingFace Transformers |
| Task | 3-class sentiment classification |

> RoBERTa is a transformer-based model developed by Facebook AI and fine-tuned specifically on Twitter/social media data — making it highly accurate for this use case.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend & UI | [Streamlit](https://streamlit.io) |
| AI / NLP Model | [HuggingFace Transformers](https://huggingface.co) |
| Deep Learning | PyTorch |
| Charts | Plotly |
| Data Handling | Pandas |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Live Demo

🔗 **[Click here to open the app](https://your-username-sentiment-analyzer.streamlit.app)**

> Replace the link above with your actual Streamlit deployment URL.

---

## 📁 Project Structure
