# EX-06-Feature-Transformation

# AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# Method For Data Tranformation
### 1. FUNCTION TRANSFORMATION
### 2. POWER TRANSFORMATION
### 3. POWER TRANSFORMATION
 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

### Developed by: Y Shavedha
### Register Number: 212221230095
### Data_To_Transform.csv:
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 
df=pd.read_csv("Data_To_Transform.csv")  
df 
df.skew() 

#Log Transformation  
np.log(df["Highly Positive Skew"])  

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])

#Square Transformation  
np.square(df["Highly Negative Skew"])

# POWER TRANSFORMATION
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df 
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df

#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show() 
df.skew()  
df 
```
### titanic_dataset.csv:
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"])  

#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  


#QUANTILE TRANSFORMATION  

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  


df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  

sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
```

# OUTPUT
## Data_To_Transform.csv:

![output](./q1.PNG)

![output](./q2.PNG)

![output](./q3.PNG)

![output](./q4.PNG)

![output](./q5.PNG)

![output](./q6.PNG)

![output](./q7.PNG)

![output](./q8.PNG)

![output](./q9.PNG)

![output](./q10.PNG)

![output](./q11.PNG)

![output](./q12.PNG)

![output](./q13.PNG)

![output](./q14.PNG)

![output](./q15.PNG)

![output](./q16.PNG)

![output](./q17.PNG)

![output](./q18.PNG)

## titanic_dataset.csv:

![output](./a1.PNG)

![output](./a2.PNG)

![output](./a3.PNG)

![output](./a4.PNG)

![output](./a5.PNG)

![output](./a6.PNG)

![output](./a7.PNG)

![output](./a8.PNG)

![output](./a9.PNG)

![output](./a10.PNG)

![output](./a11.PNG)

![output](./a12.PNG)

![output](./a13.PNG)

![output](./a14.PNG)

![output](./a15.PNG)

![output](./a16.PNG)

![output](./a17.PNG)

# RESULT

Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.