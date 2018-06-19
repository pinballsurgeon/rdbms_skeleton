    # -*- coding: utf-8 -*--
"""
Created on Sun Feb 11 07:22:20 2018

@author: dehlers
"""


import pyodbc
import numpy as np
# matplotlib.pyplot sublibrary pyplot, plots charts, python usese this library for plotting
import matplotlib.pyplot as plt
# pandas is used to import datasets
import pandas as pd
import time as tm
import difflib as diff

#conn_string = "driver={Oracle in OraClient11g_home1}; dbq='LLPRD'; uid='xxllflex'; pwd='xxllflex'"
#conn_string = ('DRIVER={Oracle in OraClient11g_home1};SERVER=PEBDBVM1.llflex.local;PORT=1571;UID=xxllflex;PWD=xxllflex')
#conn = pyodbc.connect(conn_string)

cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=ky01sq01;"
                      "Database=OracleInterface;"
                      "Trusted_Connection=yes;")

sF = 10

sSQL = ("SELECT * " 
          "FROM OPENQUERY(LLPRD,'select * "
                                 "from ( select TABLE_NAME "
                                              ",COLUMN_NAME "
                                              ",NUM_DISTINCT "
                                              ",DENSITY * 10000000 DENSITY "
                                              ",OWNER "
                                          "from ALL_TAB_COL_STATISTICS "
                                         "order by NUM_DISTINCT desc ) dehls "
                                "where rownum < %d " 
                                "')" % sF)

#cursor = cnxn.cursor()
#cursor.execute('SELECT * FROM LLPRD..ALL_TABLES where rownum < 10')
#cursor.execute("SELECT * FROM OPENQUERY(LLPRD,'select * from XXLLFLEX.XXLLF_INT_MACHINE')")
#cursor.execute("SELECT * FROM OPENQUERY(LLPRD,'select * from ( select * from ALL_TAB_COL_STATISTICS order by NUM_DISTINCT ) dehls where rownum < 100')")


def polyfit(X, y, degree):
    results = {}

    coeffs = np.polyfit(X, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(X)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    sauce = ssreg / sstot
           
    return sauce



#dataset = pd.read_csv('Social_Network_Ads.csv')
#dataset = pd.DataFrame(cnxn,sSQL)
#dataset = pd.read_sql(cnxn,sSQL)
dataset = pd.read_sql(sSQL,cnxn)


table_name_list = dataset.iloc[:, 0]
frames_dict = {}
frames_dict_INTERIOR = {}
                              
#X = dataset.iloc[:, [0,3]].values  # to make X a matrix, used bound of column indexes
#y = dataset.iloc[:, 3].values
#f = dataset.iloc[:, 3].values
        
rows, columns = dataset.shape                


#i=0
rar=0
sF=100
#i = 2   
for i in range(0,sF):
#for tname in table_name_list:  
    print "INITIAL QUERY"
    sum=i+1
    sSQL = ("SELECT * " 
          "FROM OPENQUERY(LLPRD,'select * from (select * "
          "from " + dataset.iloc[i, 4] + "." + dataset.iloc[i, 0] + " "
          "order by " + dataset.iloc[i, 1] + " desc " 
                                    ") dehls "
                               "where rownum < %d " 
                               "')" % sF)
    #print('row = %r' % (sum,))
    print sSQL
    now = tm.time()
    frames_dict[dataset.iloc[i, 0]] = pd.read_sql(sSQL,cnxn)
    #frames_dict[dataset.iloc[i, 0]]
    print frames_dict[dataset.iloc[i, 0]]
    
    print ( tm.time() - now )

    for t in range(0,sF):
            sum=t+1
            sSQL = ("SELECT * " 
                    "FROM OPENQUERY(LLPRD,'select * from (select " + dataset.iloc[t, 1] + " "
                                                          "from " + dataset.iloc[t, 4] + "." + dataset.iloc[t, 0] + " "
                                                                                #"order by " + dataset.iloc[i, 1] + " desc " 
                                                              ") dehls "
                                                              "where rownum < %d " 
                                                              "')" % sF)
            #print('row = %r' % (sum,))
            
            

            #now = tm.time()
            frames_dict_INTERIOR[dataset.iloc[t, 0]] = pd.read_sql(sSQL,cnxn)
            theloony = polyfit(frames_dict_INTERIOR[dataset.iloc[t, 0]].iloc[:,0].values,frames_dict[dataset.iloc[i, 0]].iloc[:,0].values,1)
            
            if ( theloony > 0.5 ) :
                print " "
                #print i + " " + t.__str__
                print "X=" + dataset.iloc[i, 4] + "." + dataset.iloc[i, 0] + "." + dataset.iloc[i, 1]
                print "Y=" + dataset.iloc[t, 4] + "." + dataset.iloc[t, 0] + "." + dataset.iloc[t, 1]
                print polyfit(frames_dict_INTERIOR[dataset.iloc[t, 0]].iloc[:,0].values,frames_dict[dataset.iloc[i, 0]].iloc[:,0].values,1)
                
            
    #if ( i == 3 ) :
        #print frames_dict[dataset.iloc[i, 0]].iloc[:,0]
        #X = frames_dict[dataset.iloc[i, 0]].iloc[:,0].values
    #elif ( i == 4 ) :
        #print frames_dict[dataset.iloc[i, 0]].iloc[:,0]
        #y = frames_dict[dataset.iloc[i, 0]].iloc[:,0].values


  # to make X a matrix, used bound of column indexes
  #  y = dataset.iloc[:, 4].values
                    
                    

    
    
    
    # Polynomial Regression
#def (X, y, 2):
# def polyfit (X, y, 2):

    
    
#print results
#print "RSQ'd"
#print polyfit(X,y,1)
#print sum

sTotal = dataset.count(0)

print "start"
#split matrix into test and training datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33333, random_state = 0)

print "next"
#fitting linear line for training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

print "thirdly"
#regressor.fit(X_train, y_train)
regressor.fit(X_train.reshape(1,-1), y_train.reshape(1,-1))

print "fourthly"
#y_pred = regressor.predict(X_test)

#plot results
print "throw plots"
plt.scatter(X_train.reshape(1,-1), y_train.reshape(1,-1), color = 'red')
plt.plot(X_train.reshape(1,-1), regressor.predict(X_train.reshape(1,-1)), color = 'green')
plt.title('Dan Ehls way to go!')
plt.xlabel('Years of experience')
plt.ylabel('Expereience')
plt.show()
