from flask import render_template,Flask,request
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
print(accuracy_score(y_test,pred))
  



appl=Flask(__name__)
file=open("model.pkl","rb")
file1=open("model1.pkl","rb")
file2=open("model2.pkl","rb")
file3=open("model3.pkl","rb")
file4=open("model4.pkl","rb")
file5=open("model5.pkl","rb")
random_Forest=pickle.load(file)
file.close()
random_Forest1=pickle.load(file1)
file1.close()
random_Forest2=pickle.load(file2)
file2.close()
random_Forest3=pickle.load(file3)
file3.close()
random_Forest4=pickle.load(file4)
file4.close()
random_Forest5=pickle.load(file5)
file5.close()

#random_Forest=pickle.load(file)
#file.close()



@appl.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        myDict = request.form
        Month = int(myDict["Month"])
        state= (myDict["state"])
        pred = [Month]
        #stateCall(state)
        #res=random_Forest.predict([pred])[0]
        if(state=="TAMILNADU"):
            res=random_Forest.predict([pred])[0]
        elif state=="WEST BENGAL":
            res=random_Forest1.predict([pred])[0]
        elif(state=="ORISSA"):
            res=random_Forest2.predict([pred])[0]
        elif(state=="PUNJAB"):
            res=random_Forest3.predict([pred])[0]
        elif(state=="UTTARAKHAND"):
            res=random_Forest4.predict([pred])[0]
        else:
            res=random_Forest5.predict([pred])[0]
        res=round(res,2)
        ans=knn.predict([[res]])[0]
        return render_template('result.html',Month=Month,state=state,res=res,ans=ans)
    return render_template('index.html')

if __name__ == "__main__":
    appl.run(debug=True)
