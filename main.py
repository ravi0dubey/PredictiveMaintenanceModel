from pandas_profiling import ProfileReport
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import pickle
import logging as log
log.basicConfig(filename = "test2.log",level=log.DEBUG, format='%(levelname)s (%(asctime)s %(name)s %(message)s ')



#Dataset upload function
def load_dataset():
    try:
        log.info("Proceeding with loading dataset ")
        df= pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\AIPredictivedataset.csv")
        log.info("Dataset upload complete")
        return df
    except Exception as  e:
        print("There is issue in Loading the dataset, please check the logs")
        log.exception(e)


# Function clean_Data does following
#1. In case dataset has any nan value fill it with 0
#2. Renaming the columns for better understanding
def clean_data(df1):
    try:
        df1.fillna(0)
        df1.rename(columns={'Air temperature [K]' : 'AirTemp','Process temperature [K]': 'ProcessTemp', 'Rotational speed [rpm]': "RotSpeed",
                        'Torque [Nm]': 'Torque', 'Tool wear [min]': 'ToolWear',
                        'Machine failure': 'MachFailure'}, inplace=True)
        log.info(f"cleasing of dataset {df1}done")
    except Exception as e:
        print(f"There is issue in replacing dataset{df1} null value with 0 and renaming the column")
        log.exception(e)


# Function extract_columns extract the fields and return Input and Target Value
def extract_columns(df):
    try:
        X = df[['ProcessTemp','RotSpeed','Torque','ToolWear','MachFailure','TWF','HDF','PWF','OSF','RNF']] #Input value
        Y = df[['AirTemp']] #Target Value
        return(X,Y)
        log.info(f"Extraction of columsn  {df}done")
    except Exception as e:
        print(f"There is issue in extraction of coulmns for {df} dataset")
        log.exception(e)


# Override default pandas configuration
pd.options.display.width = 0
pd.options.display.max_rows = 10000
pd.options.display.max_info_columns = 10000

# Step 1: load dataset into dataframe
df1= load_dataset()
# Step 2: Cleansing of data
clean_data(df1)

# Step 3: Create a profile report
profile1 = ProfileReport(df1)

# Step 4: Save the report in html format in the folder where project resides
profile1.to_file('report_AIPredictive_profile_test.html')


# Step 5: Extracting X and Y(Target Column values from  dataset
X,Y = extract_columns(df1)
X = sm.add_constant(X)


# Step 6: Creating a class of LinearRegression
try:
    linear1 =linear_model.LinearRegression()
except Exception as e:
    print("There is issue in creating the class")
    log.exception(e)



# Step 7 :Creating the model by placing dataset
try:
    linear1.fit(X,Y)
    log.info("Model Created Successfully")
except Exception as e:
    print("There is issue in creating the model")
    log.exception(e)

#formula of Linear Regression is Y = mx + c
# Step 8: Calculating the co-efficient value
m = linear1.coef_
# Step 9: Calculating the Intercept values
c = linear1.intercept_
print(f"Co-efficient value: {m}")
print(f"Linear Intercept value : {c}")



#  Step 9 Saving the model for future reference
try:
    file = "linear_regression.sav"
    pickle.dump(linear1,open(file,'wb'))
except Exception as e:
    print("There is issue in saving the model")
    log.exception(e)


# Step 10 : We can read the model created by us
saved_linear_model = pickle.load(open(file,'rb'))


# Step11: Loading new dataset to test the accuracy of the saved_linear_model
dataset_accuracy_test = pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\AIPredictivedataset_new.csv")

# Step 12:cleansing of dataset_accuracy_test
clean_data(dataset_accuracy_test)

#Step 13 : Extracing X_NEW and Y_NEW
X_NEW,Y_NEW = extract_columns(dataset_accuracy_test )


# Step 14: getting the accuracy of the linear model This is R2 statistics
try:
    print(f"accuracy of the model is {linear1.score(X_NEW,Y_NEW)}")
except Exception as e:
    print("There is issue in predicting the accuracy of the model")
    log.exception(e)