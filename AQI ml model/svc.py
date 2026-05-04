import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv("airpollution.csv")

df["Country"].fillna(df["Country"].mode()[0],inplace=True)
df["City"].fillna(df["City"].mode(),inplace=True)

print(df.isnull().sum())   

print(df.head())

# sns.boxplot(data=df)
# plt.show()

print(df.info())
print(df.shape)

q1=df["PM2.5 AQI Value"].quantile(0.25)
q3=df["PM2.5 AQI Value"].quantile(0.75)

iqr=q3-q1
min=q1-(1.5*iqr)
max=q3+(1.5*iqr)
filtered_data = df[(df["PM2.5 AQI Value"] >= min) & (df["PM2.5 AQI Value"] <= max)]
print(filtered_data.shape)

print(filtered_data.head(10))
print(filtered_data.columns)

filtered_data = filtered_data.dropna(subset=[
    "AQI Value","CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value",
    "Country","City",
    "AQI Category","CO AQI Category","Ozone AQI Category","NO2 AQI Category",
    "PM2.5 AQI Category"
]).reset_index(drop=True)

x=filtered_data[["CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value"]]
le=LabelEncoder()
y=le.fit_transform(filtered_data["AQI Category"])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

svc=SVC(kernel="linear")

svc.fit(x_train,y_train)

print("Test score",svc.score(x_test,y_test))
print("Train accuracy",svc.score(x_train,y_train))


print("Precision score",precision_score(y_test,svc.predict(x_test),average="macro"))
print("Recall score",recall_score(y_test,svc.predict(x_test),average="macro"))
print("F1 Score",f1_score(y_test,svc.predict(x_test),average="macro"))

s=confusion_matrix(y_test,svc.predict(x_test))
sns.heatmap(data=s,annot=True)
plt.show()

