# Social Media Sentiment Analysis with PySpark

A big data processing pipeline designed to load, clean, filter, and analyze the Stanford **Sentiment140 dataset** (1.6 million tweets) using Apache Spark and Python. 

This project demonstrates distributed data manipulation, handling balanced classes, text preprocessing, and high-frequency keyword extraction using PySpark's DataFrame API in an interactive Jupyter Notebook environment.

---

## 🚀 Features & Project Steps

1. **Environment Configuration:** Initialized a local standalone `SparkSession` with customized configurations.
2. **Big Data Loading:** Imported a 1.6M record CSV dataset optimizing for special characters using `ISO-8859-1` encoding.
3. **Exploratory Data Analysis (EDA):** Grouped data by target variables to verify class distributions.
4. **Data Filtering:** Sub-partitioned the primary dataset into distinct data pools for positive and negative sentiments.
5. **Distributed Tokenization & Text Cleaning:** Cleaned text via Regular Expressions (Regex), applied tokenization, exploded rows into standalone words, and compiled word frequencies.

---

## 📊 Dataset Insights & Results

The dataset processed is perfectly balanced, which provides a reliable foundation for downstream machine learning tasks.

### 1. Class Distribution

| Target Value | Sentiment Definition | Record Count |
| :--- | :--- | :--- |
| `0` | Negative | 800,000 |
| `4` | Positive | 800,000 |
| **Total** | **Processed Data Pool** | **1,600,000** |

### 2. High-Frequency Keywords
The text mining pipeline identified that structural and colloquial English tokens (such as `i`, `to`, `the`, `a`, `my`) dominated the global data layer. This underlines the necessity of applying a custom stop-word removal step for deeper semantic analytics.

---

## 🛠️ Technology Stack & Tools Used

* **Language:** Python 3.10
* **Big Data Framework:** Apache Spark 4.1.1 (PySpark)
* **Environment Manager:** Anaconda (Conda Virtual Environments)
* **IDE:** Jupyter Notebook
* **Version Control:** Git & GitHub

---

## 💻 Code Architecture Highlight

Below is a snippet of the single-line optimized text analysis execution block implemented during the workflow to tokenize and aggregate the 1.6 million data points:

```python
# Text transformation, token splitting, row explosion, and group aggregation
df.withColumn("w", explode(split(regexp_replace(lower(col("text")), "[^a-z\\s]", ""), "\\s+"))) \
  .filter(col("w") != "") \
  .groupBy("w") \
  .count() \
  .orderBy("count", ascending=False) \
  .show(20)
