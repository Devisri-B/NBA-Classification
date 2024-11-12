# NBA Classification
This project applies machine learning techniques to classify NBA game outcomes based on game statistics. It leverages a variety of classification models, including ensemble methods, and performs feature engineering, data preprocessing, and model evaluation to achieve accurate predictions.

## Project Overview
The objective of this project is to predict the outcome of NBA games (win/loss) using various game statistics as input features. This classification problem is tackled using multiple machine learning models, with a focus on achieving high predictive performance and understanding feature importance.

## Dataset
### Description
The dataset used for this project contains historical data on NBA games, including:
- Game-specific information (e.g., game ID, date, matchup)
- Home and away team statistics (e.g., field goals made, rebounds, assists, turnovers)
- Win/Loss outcomes for both home and away teams
- The dataset includes 55 features and spans multiple seasons of NBA games.

## Preprocessing
Key preprocessing steps include:
- Handling missing values in categorical columns (e.g., mean imputing missing Win/Loss outcomes)
- Standardization and scaling of numerical features
- Encoding of categorical features using techniques such as one-hot encoding and label encoding

## Methodology
### Feature Engineering
The project includes feature selection and engineering to optimize the predictive power of the input data. Some important transformations include:
- Standardizing numerical features to have zero mean and unit variance
- Encoding categorical features, such as team abbreviations, using one-hot encoding
### Model Selection
The following classifiers were utilized and compared:
- Naive Bayes
- Random Forest Classifier
- Multilayer Perceptron (MLP) Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier
- Ensemble Methods: Stacking and Voting Classifiers for improving model accuracy by combining predictions from multiple models
Hyperparameter tuning and cross-validation were used to optimize model performance.

## Evaluation Metrics
The models were evaluated using various metrics, including:
Accuracy,
Precision,
Recall,
F1 Score,
Confusion Matrix

## Usage
Requirements
To run this project, you need the following:
Python 3.x,
Jupyter Notebook,
NumPy,
Pandas,
Scikit-learn,
Matplotlib

## Results
The project achieves classification accuracy through a combination of baseline models and advanced ensemble methods. Comparative analysis of model performance is provided, highlighting the best-performing model and key factors influencing game outcomes.
