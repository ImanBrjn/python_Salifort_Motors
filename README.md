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
### Importinh packages
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
### Loading dataset
`Pandas` is used to read a dataset called **`HR_capstone_dataset.csv`.**  
```
# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
df0.head()
```
|satisfaction_level|last_evaluation|number_project|average_montly_hours|time_spend_company|Work_accident|left|promotion_last_5years|Department|salary|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|1|0.80|0.86|5|262|6|0|1|0|sales|medium|
|2|0.11|0.88|7|272|4|0|1|0|sales|medium|
|3|0.72|0.87|5|223|5|0|1|0|sales|low|
|4|0.37|0.52|2|159|3|0|1|0|sales|low|
## Step 2. Data Exploration (Initial EDA and data cleaning)
Now that the data has been loaded, it's time to understand the variables and clean the dataset.
```
# Gather basic information about the data
df0.info()
```
![class](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/ae9d1557-d62a-4592-85bf-fc4c10b95929)    
The info indicates that `Department` and `salary` are objects, which might be categorical variables. We will further investigate this later. Additionally, the dataset should not contain any missing values.
### Gathering descriptive statistics about the data
```
# Gather descriptive statistics about the data
df0.describe()
```
| |satisfaction_level|last_evaluation|number_project|average_montly_hours|time_spend_company|Work_accident|left|promotion_last_5years|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|count|14999.000000|14999.000000|14999.000000|14999.000000|14999.000000|14999.000000|14999.000000|14999.000000|
|mean|0.612834|0.716102|3.803054|201.050337|3.498233|0.144610|0.238083|0.021268|
|std|0.248631|0.171169|1.232592|49.943099|1.460136|0.351719|0.425924|0.144281|
|min|0.090000|0.360000|2.000000|96.000000|2.000000|0.000000|0.000000|0.000000|
|25%|0.440000|0.560000|3.000000|156.000000|3.000000|0.000000|0.000000|0.000000|
|50%|0.640000|0.720000|4.000000|200.000000|3.000000|0.000000|0.000000|0.000000|
|75%|0.820000|0.870000|5.000000|245.000000|4.000000|0.000000|0.000000|0.000000|
|max|1.000000|1.000000|7.000000|310.000000|10.000000|1.000000|1.000000|1.000000|

The description indicates:
1. All counts are the same (14999), which means there are no missing values.
2. The minimum of satisfaction_level indicates that some employees are not happy with their job. However, the mean of this column indicates that, overall, employees are roughly satisfied with their job.
3. The mean of last_evaluation indicates that, overall, employees are performing well.
4. The mean of average_montly_hours shows that employees work an average of 210 hours per month, which is higher than the normal working hours of 175 hours.
5. All employees have worked at the company for at least 2 years, indicating there are no interns in the data.
6. The rate of Work_accident is 14%. Although this rate might seem low at first glance, workplace accidents can be serious or even lethal. Hence, the HSSE (Health, Safety, Security, and Environment) department needs to come up with plans to reduce this rate.
7. In every 100 employees, 24 of them left the company. This is an alert for the company.
8. The promotion rate is low. Although the satisfaction level of employees is high, they did not receive enough promotions to encourage them.
### Renaming columns
As a data cleaning step, let's rename the columns as needed. This involves standardizing the column names so that they are all in `snake_case`, correcting any misspelled column names, and making column names more concise as needed. The `snake_case `is a variable naming convention where each word is in lower case and separated by underscores.
```
# Display all column names
df0.columns
```
![index](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/58f6fa5d-0a5a-47f0-a09e-15064e9ad088)
```
# Rename columns as needed

df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Display all column names after the update

df0.columns
```
![index2](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/bf904d20-adee-4b6f-8979-5c152c560b9c)

### Checking for missing values

We already know that there are no missing values. However, it's worth checking for any missing values anyway.
```
# Check for missing values
df0.isna().sum()
```
![missing](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/265f0e9c-ec30-4c01-80c1-1d8253c3bf43)
### Checking for duplicates
Now let's check for any duplicate entries in the data.
```
# Check for duplicates
df0.duplicated().sum()
```
`3008`   
3,008 rows contain duplicates. That is 20% of the data.
```
# Inspect some containing duplicates as needed
df0_duplicated = df0[df0.duplicated() == True]
df0_duplicated
```
| |satisfaction_level|last_evaluation|number_project|average_monthly_hours|tenure|work_accident|left|promotion_last_5years|department|salary|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|396|0.46|0.57|2|139|3|0|1|0|sales|low|
|866|0.41|0.46|2|128|3|0|1|0|accounting|low|
|1317|0.37|0.51|2|127|3|0|1|0|sales|medium|
|1368|0.41|0.52|2|132|3|0|1|0|RandD|low|
|1461|0.42|0.53|2|142|3|0|1|0|sales|low|
|...|...|...|...|...|...|...|...|...|...|...|
|14994|0.40|0.57|2|151|3|0|1|0|support|low|
|14995|0.37|0.48|2|160|3|0|1|0|support|low|
|14996|0.37|0.53|2|143|3|0|1|0|support|low|
|14997|0.11|0.96|6|280|4|0|1|0|support|low|
|14998|0.37|0.52|2|158|3|0|1|0|support|low|
