# Credit Customer Attrition Prediction

## Introduction

Most banks in the world provided credit card services. This service is quite popular because using credit card is a convenient way to pay your bill, grocery, rent, or anything you need. Credit card works the similarly to a short term loan where at the end of your billing cycle you have to pay your bill before the due date. In this project, a bank manager is disturbed with more and more customers leaving their credit card services. We would try to tackle this problem by creating a supervised learning model that can predict whether a customer will attrite or not. This model can help the manager to prevent customer attrition by giving a special attention to customer that is predicted to attrite.

## Dataset and Tools

The dataset was acquired from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). It has 23 features consist of customer profile and their credit card usage. It has 10,127 rows with no missing values or duplicates. The tools that were used are Jupyter Notebook and Python.

## Data Cleaning and Preprocessing

### Missing Values and Duplicates Handling

There is no missing value and duplicates so we just skipped the missing value and duplicates handling

### Useless Features

There are some features that we cannot use for our machine learning modelling so we dropped those features. The features that were dropped are:
- `CLIENTNUM`: client id number
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1` : this is the result of others using Naive Bayes
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2` : same with previous feature

### Label Encoding

For categorical feature that only have 2 unique values or have ordinal values we label encoded them.
- `Attrition_Flag`: `Attrited Customer` become 1 and `Existing Customer` become 0
- `Gender`: `M` become 1 and `F` become 0
- `Card_Category`: `Blue`, `Silver`, `Gold`, and `Platinum` turn into 0, 1, 2, and 3

### One-Hot Encoding

Other categorical features will be one-hot encoded. `Education_Level` and `Income_Category` are supposed to have ordinal values but both of them have `Unknown` values that is whay I decided to one-hot encoded them instead of label encoding. One-hot encoded features:
- `Education_Level`
- `Marital_Status`
- `Income_Category`

## Machine Learning Modelling

### Train Test Split

To prevent overfitting we will use train test split with a ratio of 80:20, we will use `stratify` to make sure the ratio of the target is maintaned.

### Standardizing and Oversampling with SMOTE

- The features was standardized using StandardScaler
- Because the target is imbalanced I used SMOTE to oversampled the minority target so that the class is a little bit more balance with a ratio of 0.5

### Fitting and Evaluation

I used Recall and AUC score to evaluate the models. There are several algorithm that were used and this is the result of all the algorithm using default parameter:
 
|No| model	| accuracytrain |	accuracytest | precision |	recall |	rocauc |
|---|---|---|---|---|---|---|
| 0	| LogisticRegression | 	0.876	| 0.895	| 0.657	| 0.729	| 0.828 | 
| 1	| KNeighborsClassifier | 	0.927	| 0.812	| 0.431	| 0.526	| 0.697 |
| 2	| DecisionTreeClassifier | 	1.000	| 0.929	| 0.776	| 0.788	| 0.872 | 
| 3	| RandomForestClassifier	| 1.000	| 0.952	| 0.888	| 0.803	| 0.892 |
| 4	| XGBClassifier	| 1.000	| 0.971	| 0.935	| 0.880	| 0.934 |
| 5	| Catboost |  0.998	| 0.972	| 0.950	| 0.874	| 0.933 |

Catboost got the highest Accuracy and Precision but because my preferred metrics evaluation for this kind of cases are Recall and AUC I will pick XGB for the model that I will further tuning

### Hyperparameter tuning

To tune the XGB model I will use HYPEROPT to find the best parameter. The target of this tuning was to find the combination of parameter that will gave use the best AUC score. Parameter that will be optimized are:
- `max_depth`
- `gamma`
- `reg_alpha`
- `reg_lambda`
- `colsample_bytree`
- `min_child_weight`
- `n_estimators`
- `seed`

### Refitting The Model

After finding the best parameter our model's accuracy and precision decreased but the recall and auc score increased. XGBoost before and after:
|Metrics|Before|After|
|--|--|--|
|accuracy|0.971|0.966|
|precision|0.935|0.861|
|recall|0.880|0.927|
|rocauc|0.934|0.950|

## Feature Importances
After modelling I tried to find the most important features to predict customer attrition and according to 4 models (Decision Tree, Random Forest, XGBoost, and CatBoost) is `Total_Trans_Ct` or the number of transactions made in the 12 months.

![fi](https://user-images.githubusercontent.com/20869651/191220751-0494369b-80b6-4a44-b083-464cbb9a2ab9.jpeg)


