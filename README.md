# **Capstone project: Providing data-driven suggestions for HR**

## Description and deliverables
In this capstone project, I will analyze a dataset and build predictive models to provide insights to the Human Resources (HR) department of a large consulting firm.   
For my deliverables, I will include the model evaluation (and interpretation if applicable), data visualizations directly related to the questions asked, ethical considerations, and the resources I used to troubleshoot and find answers or solutions. For This case study I'm using Python to conduct an EDA.
# **PACE stages**
![pace](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/8bc00ccd-4772-4341-b5ef-13d441fc8dcf)
## **Pace: Plan**
Considering the questions in the PACE Strategy Document to reflect on the Plan stage, I will consider the following:
### Understand the business scenario and problem
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They have collected data from employees, but now they don’t know what to do with it. They have approached me, a data analytics professional, and requested data-driven suggestions based on my understanding of the data. They have a specific question: what’s likely to make an employee leave the company?   
My goals in this project are to analyze the data collected by the HR department and build a model that predicts whether or not an employee will leave the company.   
If I can predict employees likely to quit, it might be possible to identify factors that contribute to their departure. Because finding, interviewing, and hiring new employees is time-consuming and expensive, increasing employee retention would be beneficial to the company.
### Familiarizing with the HR dataset

The dataset that T'll be using in this case study contains 15,000 rows and 10 columns for the variables listed below. 

**Note:** For more information about the data, refer to its source on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv).

Variable  |Description |
-----|-----|
satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
last_evaluation|Score of employee's last performance review [0&ndash;1]|
number_project|Number of projects employee contributes to|
average_monthly_hours|Average number of hours employee worked per month|
time_spend_company|How long the employee has been with the company (years)
Work_accident|Whether or not the employee experienced an accident while at work
left|Whether or not the employee left the company
promotion_last_5years|Whether or not the employee was promoted in the last 5 years
Department|The employee's department
salary|The employee's salary (U.S. dollars)
## Step 1. Imports
*   First let's import requaired packages.
### Import packages
```
# Import packages

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle
```
### Load dataset
`Pandas` is used to read a dataset called **`HR_capstone_dataset.csv`.**  
```
# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
df0.head()
```
	satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	Department	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
