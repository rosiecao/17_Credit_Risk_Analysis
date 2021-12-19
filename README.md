# 17_Credit_Risk_Analysis
# Analysis Overview
In this project, we use Python to build and evaluate several machine learning models to predict credit risk.
We adopted the following procedure:

- oversample the data using the RandomOverSampler and SMOTE algorithms.
- Undersample the data using the ClusterCentroids algorithm.
- Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm.
- Compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier.
We will evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

# Resources
Data Source: LoanStats_2019Q1.csv
Software: Python 3.7.9, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3

# Results (Balanced Accuracy Scores, Confusion Matrixes and Imbalanced Classification Reports)
 ## RandomOverSampler model
 
 ![image](https://user-images.githubusercontent.com/89699219/146693217-845b6187-ed2c-4d9f-9cb9-85fe88559f48.png)

The balanced accuracy score is 65%.
The high_risk precision is about 1% only with 62% sensitivity which makes a F1 of 2% only.
Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 68%.

## SMOTE model
![image](https://user-images.githubusercontent.com/89699219/146693273-94b09902-4605-46f8-8775-2860c48f7da5.png)

The results are pretty similar to the previous model.
The balanced accuracy score is 64%.
The high_risk precision is about 1% only with 63% sensitivity which makes a F1 of 2% only.
Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 66%.

## ClusterCentroids model

![image](https://user-images.githubusercontent.com/89699219/146693341-e3e3d0f9-2bc0-4b7b-af05-d9ac24671e09.png)

Here the balanced accuracy score is down to about 52%.
The high_risk precision is still 1% only with 63% sensitivity which makes a F1 of 1%.
Due to the high number of false positives, the low_risk sensitivity is only 40%.

## SMOTEENN model

![image](https://user-images.githubusercontent.com/89699219/146693412-f9878bc4-95b9-4a40-a564-5f258d419667.png)

The balanced accuracy score is about 62%.
The high_risk precision is still 1% only with 68% sensitivity which makes a F1 of only 2%.
Due to the high number of false positives, the low_risk sensitivity is 57%.

## BalancedRandomForestClassifier model

![image](https://user-images.githubusercontent.com/89699219/146693493-8e047504-e743-44e9-8ae3-c917dae8c00c.png)

The balanced accuracy score improved to about 79%.
The high_risk precision is still low at 4% only with 67% sensitivity which makes a F1 of only 7%.
Due to a lower number of false positives, the low_risk sensitivity is now 91% with 100% presicion.

## EasyEnsembleClassifier model

![image](https://user-images.githubusercontent.com/89699219/146693508-dc32de71-3b73-4310-b06c-79d74f3ffa22.png)

Now, the balanced accuracy score is high to about 93%.
The high_risk precision is still low at 7% only with 91% sensitivity which makes a F1 of only 14%.
Due to a lower number of false positives, the low_risk sensitivity is now 94% with 100% presicion.





