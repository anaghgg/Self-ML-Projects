#Read dataset
import pandas as pd
d=pd.read_csv('creditcard.csv')
x=d.iloc[:,:-1]
y=d.iloc[:,30]
#Split dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

#Use StandardScaler
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
xtrain=s.fit_transform(xtrain)
xtest=s.transform(xtest)

#Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=1)
xt,yt=sm.fit_resample(xtrain,ytrain)

#Dimensionality Reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
l=LDA()
xt=l.fit_transform(xt,yt)
xtest=l.transform(xtest)

#Train Classifier
from sklearn.ensemble import RandomForestClassifier
c = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
c.fit(xt, yt)


#Predict
ypred=c.predict(xtest)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

