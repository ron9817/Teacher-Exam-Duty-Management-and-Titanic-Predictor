from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.externals import joblib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)


@app.route('/')
def index():
    return render_template("homepage.html")

@app.route('/proj2')
def proj2():
	return render_template("index_ps.html")


@app.route('/proj1')
def proj1():
	return render_template("index.html")

@app.route('/result',methods=['GET', 'POST'])
def result():
	teachers_name=request.form['TNAME']
	teachers_name=teachers_name.split()

	exam_cordinator_n_hod_name=request.form['HOD_COR']
	exam_cordinator_n_hod_name=exam_cordinator_n_hod_name.split()

	junior=request.form['JU']
	junior=junior.split()

	senior=request.form['SE']
	senior=senior.split()

	teacher_not_available_list=request.form['NOT_AVI']
	teacher_not_available_list=teacher_not_available_list.split()

	n3=int(request.form['n3'])
	n5=int(request.form['n5'])

	room_no=request.form['room_no']
	room_no=room_no.split()

	##enter time table here so we cannot allocate to teacher having its subject

	teacher_not_available={}
	#print("Give information about teacher not available:- ")
	count=0
	tchr_a=[]
	day_a=[]
	for i in teacher_not_available_list:
		if count==0:
			tchr_a.append(i)
			count=1
		elif count==1:
			day_a.append(i)
			count=0

	for i in range(len(tchr_a)):
		tchr=tchr_a[i]
		day=day_a[i]
		#while tchr!="":
		if tchr in teacher_not_available:
		    teacher_not_available[tchr].append(day)
		else:
		    teacher_not_available[tchr]=[day]
	    #tchr=input()
	    #if tchr!="":
	    #    day=input()
	print(1)
	n6=[n5 for slot in range(n3)]
	n=n5*n3
	y=[]
	for i in teachers_name:
	    if i in exam_cordinator_n_hod_name:
	        y.append(['X' for x in range(n3)])
	    elif i in teacher_not_available:
	        k=[0 for x in range(n3)]
	        for val in teacher_not_available[i]:
	            k[int(val)-1]='X'
	        y.append(k)
	    else:
	        y.append([0 for x in range(n3)])
	data=pd.DataFrame(data=y,columns=[slot for slot in range(1,n3+1)],index=teachers_name)
	maxduty=[]
	for i in teachers_name:
	    if i in junior:
	        maxduty.append(5)
	    elif i in senior:
	        maxduty.append(3)
	    else:
	        maxduty.append(0)

	from random import randint
	counter=0
	f=1
	while n>0:
	    for i in range(len(teachers_name)):
	        for j in range(1,n3+1):
	            counter+=1
	            if counter>=400:
	                print("Total number of duties greater than available")
	                f=0
	                n=0
	                break
	            if data.iloc[i][j]!='X' and data.iloc[i][j]!=1 and maxduty[i]!=0 and n>0 and n6[j-1]>0:
	                num=randint(0, 1)
	                data.iloc[i][j]=num
	                if num==1:
	                    maxduty[i]-=1
	                    n6[j-1]-=1
	                    n-=1
	                    if n6[j-1]<=0:
	                        break
	            if n==0:
	                f=0
	        if f==0:
	            break

	for j in range(1,n3+1):
	    allocated=[]
	    for i in range(len(teachers_name)):
	        if data.iloc[i][j]==1:
	            while(True):
	                k=randint(0, n5-1)
	                if k not in allocated:
	                    allocated.append(k)
	                    data.iloc[i][j]=room_no[k]
	                    break

	data.columns=["Slot"+str(slot) for slot in range(1,n3+1)]
	print(data)
	lst=[]
	(n_n,m_m)=data.shape
	for i in range(n_n):
		lst.append(list(data.iloc[i]))
	print(2)
	columns_s=list(data.columns)
	teacher_lst=[]
	teacher_lst=list(data.index.values)
	c=0
	data_lst_o=[]
	for i in lst:
		data_lst=[]
		data_lst.append(teacher_lst[c])
		c+=1
		for j in i:
			data_lst.append(j)
		data_lst_o.append(data_lst)

	#return "Done"

	return render_template("result.html",data_lst_o=data_lst_o,columns_s=columns_s)

@app.route('/res',methods=['GET', 'POST'])
def res():
    if request.method=="POST":
        f=request.form
        pid=f['p_id']
        name=f['pname']
        pclass=int(f['pclass'])
        sex=f['sex']
        age=int(f['age'])
        fare=int(f['fare'])
        ticket=f['ticket']
        cabin=f['cabin']
        embarked=f['embarked']
        parch=int(f['parch'])
        sibsp=int(f['sibsp'])
        data=pd.DataFrame(data=[[pid,name,pclass,sex,age,fare,ticket,cabin,parch,sibsp,embarked]],columns=['PassengerId','Name','Pclass','Sex','Age','Fare','Ticket','Cabin','Parch','SibSp','Embarked'])
        lr_p=titanic('/home/ronkar/mysite/new_pickle/modellr.pkl','/home/ronkar/mysite/new_pickle/mean.pkl','/home/ronkar/mysite/new_pickle/std.pkl','/home/ronkar/mysite/new_pickle/fare_class.csv','/home/ronkar/mysite/new_pickle/lb_make.pkl')
        Y_test_model=lr_p.predict_res(data)
        return render_template("res.html",result=Y_test_model[0])
    else:
        return render_template("res.html",result=-1)
'''
@app.route('/jupyternotebook')
def jupyternotebook():
    return render_template("HOD+LR+present.html")

'''



class titanic:
    def __init__(self,model_l,mean,std,fare_class_l,lb_make_l):
        self.model = joblib.load(model_l)
        self.train_mean = joblib.load(mean)
        self.train_std = joblib.load(std)
        self.fare_class = pd.read_csv(fare_class_l)
        self.lb_make = joblib.load(lb_make_l)
        Embarked=['Embarked_C', 'Embarked_Q','Embarked_S']
        Titles=['Titles_Dr.', 'Titles_Master.', 'Titles_Miss.','Titles_Mr.', 'Titles_Mrs.', 'Titles_Other', 'Titles_Rev.']
        agestage=['agestage_old', 'agestage_teen', 'agestage_young']
        self.check=[Embarked, Titles, agestage]

    def predict_res(self, x_new):
        x_new = self.clean_data(x_new)
        x_new = self.engineer_features(x_new)
        x_new=x_new[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Family_size','FamilyId','agestage_code','Embarked_C','Embarked_Q','Embarked_S', 'Titles_Dr.', 'Titles_Master.', 'Titles_Miss.','Titles_Mr.', 'Titles_Mrs.', 'Titles_Other', 'Titles_Rev.','agestage_old', 'agestage_teen', 'agestage_young']]
        x_new = (x_new - self.train_mean)/ self.train_std
        res=self.model.predict(x_new)
        return res

    def clean_data(self, df):
        df = shuffle(df)
        df=df.reset_index(drop=True)
        df = df.drop_duplicates()
        ## fare
        fare_mask=df['Fare']==0
        fare_lst=[]
        for i in range(len(fare_mask)):
            if fare_mask[i] == True:
                fare_lst.append(self.fare_class[self.fare_class['Pclass']==df.iloc[i].Pclass].Fare[df.iloc[i].Pclass-1])
            else:
                fare_lst.append(df.iloc[i].Fare)

        df['Fare']=fare_lst
        ##missing values
        df.Cabin.fillna('N.A.',inplace=True)
        df=df.dropna()
        return df

    def engineer_features(self, df):
        df['Family_size']=df.SibSp+df.Parch+1
        df['agestage']=df.Age
        df['agestage']=df['agestage'].apply(self.stage_age)

        Name=df.Name
        name_lst=[]
        for i in Name:
            name_lst.append(i)

        name_lst1=[]
        for i in name_lst:
            name_lst1.append(i.split(','))

        titles=[]
        fname=[]
        for i in range(len(name_lst1)):
            titles.append(name_lst1[i][1].split()[0])
            fname.append(name_lst1[i][1].split()[1])
        Titles=pd.DataFrame(data=titles)
        Fname=pd.DataFrame(data=fname)
        surname=[]
        for i in range(len(name_lst1)):
            surname.append(name_lst1[i][0])
        Surname=pd.DataFrame(data=surname)
        df['Fname']=fname
        df['Surname']=surname
        df['Titles']=titles
        df.Titles.replace(['Lady.','Capt.', 'Don.','Ms.', 'Mlle.', 'the', 'Major.', 'Col.', 'Sir.', 'Jonkheer.',
       'Mme.'], 'Other', inplace=True)
        l=[]
        for i in df.Family_size:
            l.append(str(i))

        df['FamilySize_str']=l

        df['FamilyId']=df.Surname+df.FamilySize_str

        df=df.drop(['FamilySize_str'],axis=1)
        df.Parch.replace([3,4,5,6],3, inplace=True)
        df.Fare=df.Fare.apply(self.agefare_round)
        df.Age=df.Age.apply(self.agefare_round)

        #lb_make = LabelEncoder()
        df["Sex"] = self.lb_make.fit_transform(df["Sex"])
        df["agestage_code"] = self.lb_make.fit_transform(df["agestage"])
        df["FamilyId"] = self.lb_make.fit_transform(df["FamilyId"])
        df = pd.get_dummies(df, columns=['Embarked','Titles','agestage'])
        columns=df.columns
        x_val,y_val=df.shape
        for j in self.check:
            for i in j:
                if i not in columns:
                    df[i]=[0 for k in range(x_val)]
        return df
    def stage_age(self,x):
        if x>60:
            return 'old'
        elif x<60 and x>30:
            return 'young'
        else:
            return 'teen'

    def agefare_round(self,x):
        return int(x)


if __name__=='__main__':
	app.run(debug=True)