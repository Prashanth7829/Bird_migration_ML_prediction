# 🐦 Bird Migration ML Prediction

## 📌 Introduction

Bird migration is an important ecological process influenced by several factors such as weather conditions, geographical characteristics, flight behaviour, food availability, and environmental conditions.

This project uses Machine Learning to analyze bird migration data and predict whether a migration event is likely to be **Successful** or **Unsuccessful**.

The project also provides an interactive **Streamlit web application** where users can explore the dataset, view model performance, analyze features, and generate migration predictions.

---

## ❗ Problem Statement

Bird migration is influenced by many environmental, geographical, flight, and behavioural factors such as weather conditions, altitude, flight distance, food availability, flock size, and rest stops.

However, understanding how these factors collectively affect migration success can be difficult.

The main problem addressed by this project is to:

- Analyze bird migration data.
- Identify important factors associated with migration success.
- Preprocess the dataset for Machine Learning.
- Develop classification models to predict migration success.
- Compare different Machine Learning models.
- Provide an easy-to-use application for generating predictions.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze bird migration data across different species, regions, habitats, and environmental conditions.
2. Clean and preprocess the dataset for Machine Learning.
3. Perform feature engineering, encoding, and feature selection.
4. Identify important environmental, geographical, flight, and behavioural features.
5. Develop multiple Machine Learning classification models.
6. Compare different Machine Learning models for migration success prediction.
7. Evaluate model performance using suitable classification metrics.
8. Identify important features contributing to migration success.
9. Develop an interactive Streamlit application.
10. Provide single and batch prediction functionality.

---

## 🤖 Machine Learning Models

The project uses the following classification algorithms:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- Random Forest

The target variable is:

```text
Migration_Success_Num

1 → Successful Migration
0 → Unsuccessful Migration
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## 📥 Installation

### 1. Clone the Repository

Open a terminal or command prompt and run:

```bash
git clone https://github.com/Prashanth7829/Bird_migration_ML_prediction.git
```

Move into the project folder:

```bash
cd Bird_migration_ML_prediction
```

---

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

---

### 3. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

---

### 4. Install Required Libraries

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

After completing the installation, run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

If it does not open automatically, use the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 📊 Application Features

The Streamlit application provides different sections for working with the project:

### 🏠 Home

Provides an introduction to the Bird Migration ML Prediction project.

### 📊 Data Insights

Allows users to explore the bird migration dataset.

### 🔍 Feature Analysis

Provides information about the features used in the Machine Learning model.

### 📈 Model Performance

Displays the performance of the Machine Learning models.

### 🔮 Predict & Compare

Allows users to:

- Enter prediction data.
- Upload CSV files.
- Generate migration predictions.
- Compare predictions from different models.
- View prediction probabilities.

### ℹ️ About

Provides information about the project and development team.

---

## 📂 Project Structure

```text
Bird_migration_ML_prediction/
│
├── app.py
│
├── app_pages/
│   ├── about.py
│   ├── data_insights.py
│   ├── home.py
│   ├── model_performance.py
│   └── predict_compare.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── Bird_Migration_Analysis.csv
│       └── Bird_Migration_Analysis_Improved_Synthetic.csv
│
├── models/
│
├── notebooks/
│   ├── bird_migration (1).ipynb
│   └── bird_migration_EDA_MINI_PROJECT.ipynb
│
├── reports/
│
├── src/
│   ├── NLP/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── feature_analysis.py
│   ├── modeling.py
│   ├── train.py
│   └── train_random_forest.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🏗️ Project Architecture

The overall project workflow is:

```text
                    ┌──────────────────────┐
                    │   Bird Migration     │
                    │       Dataset        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Processing    │
                    │   & Preprocessing     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Feature Selection  │
                    │   & Engineering      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Machine Learning     │
                    │      Models          │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       Logistic Regression   Decision Tree     KNN
              │
              ▼
             SVM
              │
              ▼
       Gradient Boosting
              │
              ▼
        Random Forest
              │
              ▼
                    ┌──────────────────────┐
                    │  Model Evaluation    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Importance   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Migration Prediction │
                    │  & Model Comparison  │
                    └──────────────────────┘
```

---

## 🔄 Prediction Workflow

```text
User Input / CSV File
        │
        ▼
Data Validation
        │
        ▼
Feature Preprocessing
        │
        ▼
Trained ML Model
        │
        ▼
Migration Prediction
        │
        ├──────────────► Successful Migration
        │
        └──────────────► Unsuccessful Migration
```

---

## 👥 Project Team

This project was developed as a two-member team:

- **2582401 – Prashanth K**
- **2582402 – Leo Samuel Gilbert**

The project was developed collaboratively using Git and GitHub with separate development branches and a final integrated `main` branch.

---

## 🔗 GitHub Repository

Repository:

https://github.com/Prashanth7829/Bird_migration_ML_prediction

---

## ⚠️ Disclaimer

This project is developed for academic and educational purposes.

The predictions generated by the system should not be considered authoritative scientific predictions of real-world bird migration behaviour.

Real-world migration outcomes can depend on many factors that may not be represented in the dataset.

---

