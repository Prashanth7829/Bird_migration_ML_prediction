# 🐦 Bird Migration ML Prediction

An end-to-end **Machine Learning project for predicting bird migration success** using environmental, geographical, biological, temporal, flight, and behavioural features.

The project includes data preprocessing, feature analysis, multiple machine learning models, model evaluation, feature importance analysis, and an interactive **Streamlit dashboard** for prediction and comparison.

---

## 📌 Project Overview

Bird migration is influenced by several factors, including weather conditions, geographical characteristics, flight behaviour, food availability, and other environmental and biological conditions.

This project uses Machine Learning to analyze these factors and predict whether a bird migration event is likely to be **successful or unsuccessful**.

The system provides an interactive web application where users can:

- Explore the bird migration dataset
- Analyze migration-related features
- View feature importance
- Compare multiple Machine Learning models
- View model performance metrics
- Upload CSV files for batch prediction
- Generate migration success predictions
- Compare predictions and success probabilities across models

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze bird migration data.
2. Identify important factors affecting migration success.
3. Preprocess numerical and categorical features.
4. Prevent data leakage by excluding post-outcome variables.
5. Train multiple Machine Learning classification models.
6. Optimize model parameters using randomized hyperparameter search.
7. Evaluate and compare model performance.
8. Analyze feature importance.
9. Develop an interactive Streamlit application.
10. Provide single/batch prediction functionality through the application.

---

# 🧠 Machine Learning Approach

The target variable used for prediction is:

```text
Migration_Success_Num
```

where:

```text
1 → Successful Migration
0 → Failed Migration
```

The Machine Learning workflow is:

```text
                Bird Migration Dataset
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
                  Feature Selection
                         │
                         ▼
                Leakage Prevention
                         │
                         ▼
                 Train/Test Split
                         │
                         ▼
             Feature Preprocessing
             ┌───────────┴───────────┐
             │                       │
        Numerical                Categorical
        Features                  Features
             │                       │
       Imputation +             Imputation +
        Scaling                One-Hot Encoding
             │                       │
             └───────────┬───────────┘
                         ▼
                  Model Training
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
              Feature Importance
                         │
                         ▼
               Streamlit Dashboard
                         │
                         ▼
                    Prediction
```

---

# 🤖 Machine Learning Models

The project trains and compares **six classification algorithms**:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model for binary classification.

### 2. Decision Tree

A tree-based model that makes predictions using a sequence of feature-based decisions.

### 3. K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts a class based on neighboring observations.

### 4. Support Vector Machine (SVM)

A classification algorithm that finds an optimal decision boundary between classes.

### 5. Gradient Boosting

An ensemble learning method that combines multiple weak learners sequentially to improve prediction performance.

### 6. Random Forest

An ensemble of decision trees that combines multiple trees to improve generalization and reduce overfitting.

---

# 🔧 Model Optimization

The project uses:

```text
RandomizedSearchCV
```

for hyperparameter optimization.

The training configuration includes:

- 3-fold cross-validation
- Randomized hyperparameter search
- 8 parameter combinations per model
- `random_state = 42`
- 80% training data
- 20% testing data

The primary optimization metric used during model selection is:

```text
ROC-AUC
```

---

# 📊 Model Evaluation

The trained models are evaluated using multiple classification metrics, including:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The application provides a dedicated **Model Performance** page to compare the models.

---

# 🧬 Features Used for Prediction

The project uses approved features from several categories.

## Biological

- `Species`

## Geographical

- `Region`
- `Habitat`
- `Origin`
- `Start_Latitude`
- `Start_Longitude`
- `End_Latitude`
- `End_Longitude`

## Migration Motivation

- `Migration_Reason`

## Environmental

- `Weather_Condition`
- `Temperature_C`
- `Wind_Speed_kmph`
- `Humidity_%`
- `Pressure_hPa`
- `Visibility_km`

## Flight Characteristics

- `Flight_Distance_km`
- `Flight_Duration_hours`
- `Average_Speed_kmph`
- `Max_Altitude_m`
- `Min_Altitude_m`

## Behavioural

- `Rest_Stops`
- `Predator_Sightings`
- `Migrated_in_Flock`
- `Flock_Size`
- `Food_Supply_Level`

## Temporal

- `Migration_Start_Month`
- `Migration_End_Month`

---

# 🛡️ Data Leakage Prevention

The project deliberately does **not** use every column in the dataset for prediction.

Variables that contain target information, post-outcome information, redundant representations, or analysis-only information are excluded from the Machine Learning feature set.

Examples include:

```text
Migration_Success
Migration_Success_Num
Migration_Interrupted
Interrupted_Reason
Recovery_Location_Known
Recovery_Time_days
Nesting_Success
Tag_Battery_Level_%
Tracking_Quality
Food_Supply_Num
Flock_Size_Band
Flock_Size_Num
Flight_Distance_Category
Speed_Category
Altitude_Category
Temperature_Category
Wind_Speed_Category
Visibility_Category
Humidity_Category
Cluster
Migration_Behavior_Cluster
Path
```

This helps ensure that the model uses information that is available before or during migration rather than information that may only become available after the outcome.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

The main application is launched through:

```text
app.py
```

## Application Pages

### 🏠 Home

Provides an introduction to the Bird Migration ML Prediction project.

### 📊 Data Insights

Provides data exploration and visual analysis of the migration dataset.

### 🔍 Feature Analysis

Allows users to explore the importance and relationships of migration-related features.

### 📈 Model Performance

Displays performance metrics and comparisons between the six Machine Learning models.

### 🔮 Predict & Compare

Allows users to:

- Enter prediction data
- Download a CSV prediction template
- Upload prediction CSV files
- Generate predictions
- Compare predictions from different models
- View success probabilities

### ℹ️ About

Provides information about the project and development team.

---

# 📂 Project Structure

```text
Bird_migration_ML_prediction/
│
├── .gitignore
│
├── .streamlit/
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
│   └── processed/
│       └── Bird_Migration_Analysis.csv
│
├── models/
│   └── Saved Machine Learning model artifacts
│
├── notebooks/
│   └── Jupyter notebooks
│
├── reports/
│   └── Model and analysis reports
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
├── README.md
│
└── requirements.txt
```

---

# 📁 Important Files

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit application entry point |
| `app_pages/` | Individual Streamlit application pages |
| `src/modeling.py` | Defines features, target, preprocessing and prediction helpers |
| `src/train.py` | Trains and evaluates the six Machine Learning models |
| `src/train_random_forest.py` | Random Forest training workflow |
| `src/feature_analysis.py` | Feature analysis functionality |
| `src/dashboard.py` | Dashboard-related functionality |
| `data/processed/` | Processed dataset used for Machine Learning |
| `models/` | Saved trained model pipelines and evaluation artifacts |
| `notebooks/` | Jupyter notebooks used during development and analysis |
| `reports/` | Generated reports and analysis outputs |
| `requirements.txt` | Python dependencies |

---

# ⚙️ Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Web Application

- Streamlit

## Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Prashanth7829/Bird_migration_ML_prediction.git
```

Move into the project directory:

```bash
cd Bird_migration_ML_prediction
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Step 1 — Activate the Virtual Environment

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Step 2 — Train the Models

Run:

```bash
python -m src.train
```

This trains and evaluates all six Machine Learning models and generates the required model and evaluation artifacts.

---

## Step 3 — Start the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

# 🧪 Batch Prediction

The **Predict & Compare** page provides a downloadable CSV template for testing predictions.

The uploaded CSV must contain **all required Machine Learning input features**.

The required columns include:

```text
Species
Region
Habitat
Origin
Migration_Reason
Weather_Condition
Migration_Start_Month
Migration_End_Month
Migrated_in_Flock
Food_Supply_Level
Start_Latitude
Start_Longitude
End_Latitude
End_Longitude
Temperature_C
Wind_Speed_kmph
Humidity_%
Pressure_hPa
Visibility_km
Flight_Distance_km
Flight_Duration_hours
Average_Speed_kmph
Max_Altitude_m
Min_Altitude_m
Rest_Stops
Predator_Sightings
Flock_Size
```

### Important

The column names must match the expected feature names.

For example:

```text
Wind_Speed_kmph
Humidity_%
Average_Speed_kmph
```

should not be renamed to:

```text
Wind Speed
Humidity
Average Speed
```

Extra columns in the uploaded CSV can be ignored by the prediction pipeline, but all required model features must be present.

---

# 📥 Prediction Output

After uploading a valid CSV, the application generates prediction results.

The prediction output includes:

- Prediction from each saved Machine Learning model
- Migration success probability
- Model comparison information

This allows users to evaluate how different classifiers behave on the same migration input.

---

# 🔬 Preprocessing

The project uses a shared preprocessing pipeline.

## Numerical Features

Numerical features are processed using:

```text
Median Imputation
        ↓
Standard Scaling
```

## Categorical Features

Categorical features are processed using:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

Unknown categorical values are handled using:

```text
handle_unknown = "ignore"
```

The preprocessing steps are included inside the Machine Learning pipelines to ensure consistent transformation during training and prediction.

---

# 📈 Feature Importance

The project also analyzes the importance of input features.

Different approaches are used depending on the model type:

- Tree-based feature importance
- Model coefficients for linear models
- Permutation importance for KNN

Categorical one-hot encoded features are grouped back into their original source features for easier interpretation.



# 🌿 GitHub Collaboration

The project is maintained using Git and GitHub with a branch-based collaboration workflow.

```text
                         main
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
   Prashanth Development       Leo Development
             │                         │
             │                         │
             └────────────┬────────────┘
                          │
                    Pull Requests
                          │
                          ▼
                         main
                  Final Integrated Code
```

The `main` branch represents the stable and integrated version of the project.

Development work can be performed in separate branches and merged into `main` through Pull Requests.

---

# 🔐 Data and Privacy

This project is intended for academic and educational purposes.

The application does not intentionally collect personally identifiable information.

---

# ⚠️ Disclaimer

This project is an academic Machine Learning application developed for educational and demonstration purposes.

The predictions generated by this system should not be considered authoritative scientific predictions of real-world bird migration behaviour.

Real-world migration outcomes can depend on many factors that may not be represented in the dataset.

---

# 🚀 Future Enhancements

Possible future improvements include:

- Integration with real-time weather APIs
- Integration with real-world bird tracking datasets
- Real-time migration monitoring
- Geographical migration maps
- Advanced hyperparameter optimization
- Additional Machine Learning algorithms
- Deep Learning-based prediction
- Automated model retraining
- Cloud deployment
- Improved prediction visualization
- More advanced model explainability
- Real-time environmental data integration

---

# 📚 Academic Context

This project demonstrates an end-to-end Machine Learning workflow including:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Exploratory Data Analysis
      ↓
Machine Learning
      ↓
Hyperparameter Optimization
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Streamlit Deployment
      ↓
Prediction & Comparison
```

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

# ⭐ Acknowledgements

We would like to thank our faculty and institution for their guidance and support throughout the development of this project.

---

## 🔗 Repository

**GitHub:**  
https://github.com/Prashanth7829/Bird_migration_ML_prediction