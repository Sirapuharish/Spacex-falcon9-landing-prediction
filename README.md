🚀 SpaceX Falcon 9 Landing Prediction
Overview

This project predicts whether the Falcon 9 first stage will successfully land based on launch features such as payload mass, orbit type, launch site, booster serial number, and other encoded variables. The goal is to understand what factors influence booster recovery and assist in cost estimation.

Project Goal

To build a machine learning classification model that predicts landing success (1 = landed, 0 = failed).
The deployed Flask app uses Logistic Regression.

Repository Contents

app.py — Flask web application

spacex_model.pkl — trained Logistic Regression model

model_columns.pkl — feature list used during model training

templates/ — HTML files (web UI)

static/ — CSS files

Jupyter notebooks for data collection, wrangling, EDA, and model training

Technologies / Libraries Used

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

preprocessing

train_test_split

GridSearchCV

LogisticRegression

SVC

DecisionTreeClassifier

KNeighborsClassifier

Flask

Methodology

Collected SpaceX data using API and web scraping

Cleaned and processed the dataset

Performed EDA using visualization and SQL

Trained multiple ML models and compared performance

Selected Logistic Regression and deployed it via Flask

Model Performance

Logistic Regression

Jaccard Score: 0.80

F1 Score: 0.7777

Other models were tested (SVM, Decision Tree, KNN), but Logistic Regression performed best for deployment.
