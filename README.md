SpaceX Falcon 9 Landing Prediction

Overview
This project predicts whether the Falcon 9 first stage will successfully land based on launch features such as payload mass, orbit type, launch site, booster details, and other encoded parameters. The purpose of the project is to analyze the factors that influence landing success and build a machine learning model that can classify the outcome.

Project Goal
The goal is to create a machine learning classification model to predict landing success (1 or 0). Logistic Regression was selected as the final model and integrated into a Flask web application.

Repository Contents
app.py – Flask application file
spacex_model.pkl – trained Logistic Regression model
model_columns.pkl – feature names used for training
templates – HTML interface files
static – CSS files
Jupyter notebooks – used for data collection, preprocessing, EDA, and model training

Technologies and Libraries Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn (preprocessing, train_test_split, GridSearchCV, LogisticRegression, SVC, DecisionTreeClassifier, KNeighborsClassifier)
Flask

Methodology
Collected and combined SpaceX launch data using API and web scraping.
Performed data cleaning, feature engineering, and encoding.
Conducted exploratory data analysis using visualizations and SQL queries.
Trained multiple machine learning models including Logistic Regression, SVM, Decision Tree, and KNN.
Evaluated all models and selected Logistic Regression based on performance and reliability.
Deployed the final model through a Flask web interface.

Model Performance
Logistic Regression achieved a Jaccard score of 0.80 and an F1 score of 0.77 during evaluation.
