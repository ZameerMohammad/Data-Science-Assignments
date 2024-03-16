

import pandas as pd
import numpy as np

df = pd.read_csv("50_Startups.csv")
df

df.shape
df.dtypes
list(df)

df.head()

# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['State']=LE.fit_transform(df['State'])
# splitting 
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
Y = df['Profit']

 #-------- Standaraization ---------- # 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)
ss_x
ss_x = pd.DataFrame(ss_x)
ss_x

# ---------- Model Fitting ---------#
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(ss_x,Y)
ss_Y_pred=LR.predict(ss_x)

#-----------Metrics-------------#
from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using standard scalar r^2 score**")
print("R square", r2_score(Y,ss_Y_pred).round(2))



#---------min-max scalar---------#

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X=pd.DataFrame(MM_X)
MM_X

# ---------- Model Fitting ---------#
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(MM_X,Y)
MM_Y_pred=LR.predict(MM_X)

#-----------metrics---------------#
from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using min-max scalar r^2 score**")
print("R square", r2_score(Y,MM_Y_pred).round(2))




#====================== Toyota Corolla ======================#

import pandas as pd
import numpy as np

df = pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
df

df.shape
df.dtypes
list(df) 

df.head()

X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
Y = df['Price']

#----------------Standardization---------------------#
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_X = ss.fit_transform(X)
ss_X = pd.DataFrame(ss_X)
ss_X

#--------------Min-max Scaler------------------#

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X=pd.DataFrame(MM_X)
MM_X

#----------------MultiLinear Regression Using Standardization--------------------#

from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(ss_X,Y)
ss_Y_pred=LR.predict(ss_X)
ss_Y_pred


from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using standard scalar r^2 score**")
print("R square", r2_score(Y,ss_Y_pred).round(2))


#-------------------MultiLinear Regression Using Min-Max Scalar-------------------#

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X=pd.DataFrame(MM_X)
MM_Y_pred=LR.predict(MM_X)
MM_Y_pred



from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using Min-Max scalar r^2 score**")
print("R square", r2_score(Y,MM_Y_pred).round(2))















