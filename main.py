from pandas_profiling import ProfileReport
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

#load dataset into dataframe
df= pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\AIPredictivedataset.csv")

#in case dataset has any nan value fill it with 0
df.fillna(0)

# print(df)

# Override default pandas configuration
pd.options.display.width = 0
pd.options.display.max_rows = 10000
pd.options.display.max_info_columns = 10000

# #Create a profile report
# profile1 = ProfileReport(df)

# #this will create a report in html format in the folder where project resides
# profile1.to_file('AIPredictive_profile_test.html')
df.rename(columns={'Air temperature [K]' : 'AirTemp','Process temperature [K]': 'ProcessTemp', 'Rotational speed [rpm]': "RotSpeed",
                    'Torque [Nm]': 'Torque', 'Tool wear [min]': 'ToolWear',
                    'Machine failure': 'MachFailure'}, inplace=True)
# print(df.head(5))
X= df[['ProcessTemp','RotSpeed','Torque','ToolWear','MachFailure','TWF','HDF','PWF','OSF','RNF']]
Y = df[['AirTemp']]

# print(X)
# print(Y)
# print(x1)
X = sm.add_constant(X)
print(X)

linear1 =linear_model.LinearRegression()
linear1.fit(X,Y)

#Calculating the co-efficient value
m = linear1.coef_
#Calculating the Intercept values
c = linear1.intercept_
#formula is Y = mx + c

print(f"Co-efficient value: {m}")
print(f"Linear Intercept value : {c}")
#
