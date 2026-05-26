# 🐦 Twitter Sentiment Analysis using BERT

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project performs **Sentiment Analysis** on Twitter (X) data by first scraping real-time tweets using **Selenium**, then applying a **fine-tuned BERT model** to classify the sentiment of each tweet as **Positive**, **Neutral**, or **Negative**. The workflow includes data preprocessing, exploratory data analysis (EDA), and visualization of results.

The goal is to demonstrate an end-to-end Natural Language Processing (NLP) pipeline for analyzing public opinion on social media.

## ✨ Key Features

- **Web Scraping:** Uses Selenium to scrape tweets from a specified trending page on X (formerly Twitter) with anti-detection measures.
- **Data Preprocessing:** Cleans tweet text by removing URLs, mentions, hashtags, and special characters.
- **BERT-based Sentiment Analysis:** Leverages a pre-trained and locally fine-tuned BERT model (`bert_finetuned_local`) for three-class sentiment classification.
- **Data Visualization:** Generates count plots and word clouds to visualize sentiment distribution and common keywords.
- **Modular Code:** Organized into reusable functions and classes.
