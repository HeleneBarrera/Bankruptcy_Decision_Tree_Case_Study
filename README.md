# Bankruptcy Prediction
The objective of this project is to use Random Forest and XGBoost to accurately predict a company's likelihood to go bankrupt based on financial information provided in .arff file formats. The financial data consists of a set of 64 attributes, each containing data such as net profits by total assets or total assets by total liabilities.

## Data Evaluation/Engineering
During the data evaluation process, we discovered that the data needed some cleaning before it could be used in the model. We identified 21 variables with more than 95% correlation and removed them from the dataset. We also found that there were missing values in every line of data, so we used a SimpleImputer from Sklearn to impute these missing values in the remaining features.

## Methods
We used both Random Forest and XGBoost models. Since both are decision tree-based algorithms that take different approaches to classifying data, it was interesting to compare the results side-by-side. Given the business objective, decision trees in general are an excellent approach to this classification problem because they are somewhat interpretable even for a large number of features and they provide a list of important features. 

Random Forest tends to perform better on balanced and preprocessed data, while XGBoost is a better option for unbalanced data. Random Forest works in parallel by sampling the features and data across multiple trees and making a final prediction. XGBoost is an iterative process that weights iterations with incorrect classifications heavier, ultimately giving more weight to the smaller class and less weight to the larger class.

To determine which approach is more useful, we used a variety of evaluation metrics such as accuracy, precision, and recall to compare the performance of the two models.

## Results
We ran two Random Forest models: a basic model with a balanced subsample, and a tuned model using GridSearchCV with 5-folds and refit based on the accuracy score. The tuned model had an accuracy of .96, with a precision and recall of .95 and .96 respectively. This was a slight improvement on the untuned model.

We ran two XGBoost models: a basic model, and a tuned model using RandomSearchCV with 5-folds. The tuned model had an accuracy of .96, with a precision and recall of .97 and .96 respectively. This model performed slightly better than the tuned Random Forest model.

The top 3 features of importance were:
Attr34: Operating Expenses/Total Liabilities
Attr27: Profit on Operating Activities/Financial Expenses
Attr5: [(Cash + Short-Term Securities + Receivables - Short-Term Liabilities) / (Operating Expenses - Depreciation)] * 365

These features provide insight into a company's financial health and ability to pay its bills, which are important factors in predicting bankruptcy.

## Business Case
We propose that decision makers for the company use the general XGBoost model we created based on the accuracy score to fit their needs. While there will be a small percentage of incorrect predictions, decision makers can be confident in the features being used for these predictions.

When moving forward, decision makers should consider the top 3 features of importance and potentially look further into those attributes to understand their impact. We also recommend exploring the possibility of tuning the results to focus on correctly classifying 99.9% or more of companies that will go bankrupt, even if it means creating false positives.

Overall, these models are highly interpretable and can be customized to meet business needs. While no model is perfect, decision trees can be used to build a variety of models that improve understanding of the existing problem and provide strong predictive power for the future.