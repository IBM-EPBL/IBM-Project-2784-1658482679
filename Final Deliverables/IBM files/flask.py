import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "6LfbCZ72apZnwHTs9njb0wj2UlpLWNDPIk8wEN8ayRAH"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]


from flask import render_template,Flask,request
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.naive_bayes import GaussianNB
dt = pd.read_csv(r"C:/Users/NIVEDITHA/Downloads/Crop_recommendation.csv")
  
# Create feature and target arrays
train=dt['rainfall']
target=dt['label']
train=np.array(train)
target=np.array(target)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             train,target, test_size = 0.3, random_state=1)
  
knn = GaussianNB()
  
knn.fit(X_train.reshape(-1,1), y_train)
pred=knn.predict(X_test.reshape(-1,1))
 


appl=Flask(__name__)


#random_Forest=pickle.load(file)
#file.close()



@appl.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        myDict = request.form
        Month = int(myDict["Month"])
        state= (myDict["state"])
        xtest=[[Month]]
        if (state=="TAMILNADU"):
            payload_scoring = {"input_data": [{"field": [['Month']], "values": xtest}]}
            response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/b9911f2a-8c37-43ff-86ca-5857254f4cb3/predictions?version=2022-11-21', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
            print(response_scoring)
            predic=response_scoring.json()
            predic2=round(predic['predictions'][0]['values'][0][0],2)
            print(predic2)        
        else:
            payload_scoring = {"input_data": [{"field": [['Month']], "values": xtest}]}
            response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/6be082f6-3e57-4824-9a76-4bbbdea23f45/predictions?version=2022-11-21', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
            print(response_scoring)
            predic=response_scoring.json()
            predic2=round(predic['predictions'][0]['values'][0][0],2)
            print(predic2) 
        
        ans=knn.predict([[predic2]])[0]
        return render_template('result.html',Month=Month,state=state,res=predic2,ans=ans)
    return render_template('index.html')

if __name__ == "__main__":
    appl.run(debug=True)










    