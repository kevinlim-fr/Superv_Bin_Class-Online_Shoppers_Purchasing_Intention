# ESILV Python for Data Analysis Project - Online Shoppers Purchase Intention
This project is an end of semester project at ESILV, the goal is to analyse a dataset, build a predictive model and deploy an API. The assigned dataset is the UCI Online Shoppers Purchase Intention.

# Dataset Presentation
UCI Online Shoppers Purchasing Intention Dataset Data - https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
The goal of the dataset analysis is to analyse a set of online shopping session that leads (or not) to a purchase.
The dataset contain 12 330 session with 10 422 of the session that doesn't lead to a purchase and the rest (1 908) of the sessions leads to a purchase.

# The Classification Problem
The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute is used as class label. The dataset is clean, there is no missing values but it is unbalanced. There is a risk of bias, so the analysis have to take the unbalanced factor into consideration.

## Classification algorithms
I have selected a few Classification algorithm based on the accuracy of the prediction, but also the F-score as the dataset is unbalanced.
I performed a comparison of 9 Classification algorithm. (Naive Bayes, Logistic Regression, K Nearest Neighbour, Support Vector Machine, Decision Tree, Stochastic Gradient Descent, Linear Discriminant Analysis, Gradient Boosting, Random Forest)

## Hyperparameters optimization (Grid Search)
Then selected the 4 best performing ones (Gradient Boosting, Stochastic Gradient Descent, Random Forest, Decision Tree) and made 4 grid search to optimise the hyperparameters.
After the hyperparmeter optimization, I chose to select the 2 best algorithm (Random Forest and Decision Tree) and make a VotingClassifier model for prediction.

# API
I created a rest API that listen to GET request. Any user can predict a Purchase (or Not) by inputing the variable values as listed in the dataset. The API will then answer if this session leads to a Purchase or not.
(example: curl -X GET http://127.0.0.1:5000/ -d '0,0,0,0,1,0,0.2,0.2,0,0,Feb,1,1,1,1,Returning_Visitor,FALSE')
Response: Not Purchased
