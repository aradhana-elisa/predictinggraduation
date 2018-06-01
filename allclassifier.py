# -*- coding: utf-8 -*-
"""

@author: Aradhana Elisa
"""
#Tasks
# 1: If the student graduates within 4 years
# 2: If the student graduates within 5 years
# 3: If the student graduates within 6 years

# Importing the libraries
# Handle table-like data and matrices
import numpy as np
import pandas as pd
# Modelling Helpers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn import metrics
from math import sqrt
# Handle table-like data and matrices
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Visualisation
import matplotlib.pylab as plt
import seaborn as sns


#Values for the models that can be used to fit the data
clfs = {'LR': LogisticRegression(), 
        'DT': DecisionTreeClassifier(criterion = 'gini', random_state = None),
        'KNN' : KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2, n_jobs=-1),
        'RF' : RandomForestClassifier(n_estimators = 50, criterion = 'gini', random_state = None,max_features= 'sqrt')
    }

#Take input from the as to for which task are we predicting
task = input("Enter the Task: ")
# Importing the dataset
org_data = pd.read_csv('datafiles/dataset.csv')


#Partition the dataset based on the task selection the user makes
if(task == 1): 
    org_data = org_data[org_data.Matric <= 20134]
    org_data = org_data.drop(['Degree5yr', 'Degree6yr', 'Matric'], axis=1)
    col_drop = 'Degree4yr'
elif(task == 2):
    org_data = org_data[org_data.Matric <= 20124]
    org_data = org_data.drop(['Degree4yr', 'Degree6yr', 'Matric'], axis=1)
    col_drop = 'Degree5yr'    
elif(task == 3):
    org_data = org_data[org_data.Matric <= 20114]
    org_data = org_data.drop(['Degree4yr', 'Degree5yr', 'Matric'], axis=1)
    col_drop = 'Degree6yr'   
else:
    print ("Please select a task from 1, 2 or 3")

#Changing the format of the period from 20122 to 2012
org_data["Period"] = org_data["Period"].astype(str).str[:4].astype(int)

#Splitting the dataset into different dataframes as per the Student Level
for Std_level, test in org_data.groupby('Std_level'):
    if ( 1 in org_data.Std_level):
        data_freshmen = org_data[org_data.Std_level == 1]
        if (2 in org_data.Std_level):
            data_sophomore = org_data[org_data.Std_level == 2]
            if (3 in org_data.Std_level):
                data_junior = org_data[org_data.Std_level == 3]
                if(4 in org_data.Std_level):
                    data_senior = org_data[org_data.Std_level == 4]
    else:
        data_senior = org_data[org_data.Std_level == 4]
    
#delete extra information
del test
del Std_level 

#Let user enter the student level for which he wants the data
student_level = input("Enter the Student Level: ")

if (student_level == 1):
    std_level_label = 'freshmen'
    dataset = data_freshmen    
elif (student_level == 2):
    std_level_label = 'sophomore'
    dataset = data_sophomore
elif (student_level == 3):
    std_level_label = 'junior'
    dataset = data_junior
else:
    std_level_label = 'senior'
    dataset = data_senior

del student_level

#The dependent variable form the dataset
y = dataset.iloc[:, 30].values 

#Encoding the dummy variables into numercial data using pd_dummies
dataset = pd.get_dummies(dataset, columns=["Academic_standing", "College_name", "Enr_status", "Ethnicity", "Residence"], prefix=["acad", "college", "enr", "eth", "res"])
dataset = dataset.drop([col_drop, 'Std_level', 'Id'], axis=1)

#List of the Independent variables from the dataset
X = dataset.iloc[:, 0:58].values

# Taking care of missing data
# URM, FGS - by mode
imputer = preprocessing.Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 11:13])
X[:, 11:13] = imputer.transform(X[:, 11:13])

#SAT, HSGPA, TRGPA - by mean
imputer2 = preprocessing.Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X[:, 8:11])
X[:, 8:11] = imputer2.transform(X[:, 8:11])

#For Scaling the dataset
# calculate column means of all the columns
def column_means(X):
	means = [0 for i in range(len(X[0]))]
	for i in range(len(X[0])):
		col_values = [row[i] for row in X]
		means[i] = sum(col_values) / float(len(X))
	return means

# calculate column standard deviations
def column_stdevs(X, means):
    stdevs = [0 for i in range(len(X[0]))]
    for i in range(len(X[0])):
        variance = [pow(row[i]-means[i], 2) for row in X]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(X)-1))) for x in stdevs]
    return stdevs
# standardize dataset
def standardize_dataset(X, means, stdevs):
    for row in X:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]
# Estimate mean and standard deviation
means = column_means(X)
stdevs = column_stdevs(X, means)
standardize_dataset(X, means, stdevs)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=None)


def FeatureSelection(): 
    """
    Purpose
    ----------
    Function helps to see how many optimal features are required and also genertaes a 
    bar chart detailing variable importance for CART model

    Returns
    ----------
    The optimal number of features for each criteria
    A graph of the number of features vs the AUC for each splits 
    The list of all the features with the mean entropy values  
    Also the value of the AUC after the model with the new features
    """    
    #All the features in my dataset
    features = dataset.columns.values[0:58]
    #Fit the Random Forest Model
    classifier = RandomForestClassifier(n_estimators = 50, criterion = 'gini', random_state = None, max_features= 'sqrt')
   
    #The estimator is trained on the initial set of features
    selector = RFECV(estimator=classifier, step=1, cv=10, scoring='accuracy')
    selector.fit(X_train, y_train)
    
    #Printing the number of features from the RFECV
    print('The optimal number of features is {}'.format(selector.n_features_))
    #support has the mask of selected features
    features = [f for f,s in zip(features, selector.support_) if s]
    print('The selected features are:')
    print ('{}'.format(features))
    #Plotting the graph for the number of features and AUC
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (roc auc)")
    #grid_scores gives the cross-validation scores for the particular iteration
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.grid()
    strtask = str(task)
    plt.savefig(std_level_label +" Optimal features " +strtask)
    plt.show()
    
    
    #Importance of the features obtained by the importance attribute , the least importnace are pruned from the datset
    X_train_new = selector.transform(X_train)
    classifier2 = RandomForestClassifier(n_estimators = 50, max_features= 'sqrt', criterion = 'gini', random_state = None)
    classifier2.fit(X_train_new, y_train)
    
    importances_rf = classifier2.feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1]
    feature_space = []
    
    for f in range(len(features)):
        i = f
        print("%d. The feature '%s' has a Mean Decrease in Gini of %f" % (f + 1, features[indices_rf[i]], importances_rf[indices_rf[f]]))
        feature_space.append(features[indices_rf[i]])
   
    
    X_test_new = selector.transform(X_test)
    #y_test_new = selector.transform(y_test)
    y_pred_new = classifier2.predict(X_test_new)
    y_real_new = classifier2.predict_proba(X_test_new)
    #computing the confusion matrix from the test and predicted values
    confusion = metrics.confusion_matrix(y_test, y_pred_new)
    sns.heatmap(confusion,annot=True,fmt="d",cmap="YlGnBu",cbar=False)
    #[row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    #Printing the accuracy, error and the classification report for the model
    accuracy = ((TP + TN) / float(TP + TN + FP + FN))
    print 'Accuracy is:', accuracy
    error = ((FN + FP) / float(TP + TN + FP + FN))
    print 'Error rate is: ', error
    classification_reports = metrics.classification_report(y_test, y_pred_new, digits=4)
    print ("Classification Report: ")
    print (classification_reports)
    #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    logit_roc_auc = metrics.roc_auc_score(y_test,  y_real_new[:,1])
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_real_new[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label= 'ROC curve FS (area = %0.3f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc="lower right")
    strtask = str(task)
    plt.savefig('Graph after feature selection '+strtask)
    plt.show()
   

        
    #Generating the tree for the variables from the new train set - only the first 5 variables
    from sklearn import tree
    i_tree = 0
    for tree_in_forest in classifier2.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, max_depth =4)
        i_tree = i_tree + 1  



        
def classification(models=['LR'], doKfold=False): 
    """
    Purpose:
    ----------
    Prints the classification metric and ROC graph for the selected model

    Parameters:
    ----------
    models: Name of model you want the daa to fit
    doKfold: boolean to perform k-fold cross validation or not

    Returns:
    ----------
    The classification report, confusion matrix and the ROC curve for the data 
    Also, outputs the Precision, recall, f1score, AUC, tp, tn, fp, fn for the k-fold validations
    """
    
    global y_pred
    for ix,clf in enumerate([clfs[x] for x in models]):
        print clfs[x]
        #the classifier on the based of the model selection
        classifier = clfs[x]
        #fit the model with the training data
        classifier.fit(X_train, y_train)
        #Predicting the y value
        y_pred = classifier.predict(X_test)
        #Predicting the y-probability values
        y_real = classifier.predict_proba(X_test)
        #computing the confusion matrix from the test and predicted values
        confusion = metrics.confusion_matrix(y_test, y_pred)
        sns.heatmap(confusion,annot=True,fmt="d",cmap="YlGnBu",cbar=False)
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        #Printing the accuracy, error and the classification report for the model
        accuracy = ((TP + TN) / float(TP + TN + FP + FN))
        print 'Accuracy is:', accuracy
        error = ((FN + FP) / float(TP + TN + FP + FN))
        print 'Error rate is: ', error
        classification_reports = metrics.classification_report(y_test, y_pred, digits=4)
        print ("Classification Report: ")
        print (classification_reports)
        #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        logit_roc_auc = metrics.roc_auc_score(y_test,  y_real[:,1])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_real[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label= models[ix]+' ROC curve (area = %0.3f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend(loc="lower right")
        plt.savefig( 'figures/normal/'+col_drop+' '+std_level_label+' '+models[ix]+' graph')
        plt.show()
        
        #k-fold validation, you can change the value of the fold from the cv value in the cross_validate function
        if doKfold:
            from sklearn.model_selection import cross_validate
            from sklearn.metrics import make_scorer
            #Compute the true positives, true negatives, false postives and the false negatives for each fold and display the mean.
            def tp(y_test, y_pred): return metrics.confusion_matrix(y_test, y_pred)[1, 1]
            def tn(y_test, y_pred): return metrics.confusion_matrix(y_test, y_pred)[0, 0]
            def fp(y_test, y_pred): return metrics.confusion_matrix(y_test, y_pred)[1, 0]
            def fn(y_test, y_pred): return metrics.confusion_matrix(y_test, y_pred)[0, 1]
            
            cf_scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),'fp' : make_scorer(fp), 'fn' : make_scorer(fn)}
            classifier2 = clfs[x]
            print clfs[x]
            #results = cross_validate(classifier2,  X_std, y, cv=10,scoring=cf_scoring)
            results = cross_validate(classifier2,  X, y, cv=10,scoring=cf_scoring)
            print('K-fold cross-validation results:')
            print classifier.__class__.__name__+" average TP: %.2f (+/-%.3f)" % (results['test_tp'].mean(),results['test_tp'].std())
            print classifier.__class__.__name__+" average TN: %.2f (+/-%.3f)" % (results['test_tn'].mean(),results['test_tn'].std())
            print classifier.__class__.__name__+" average FP: %.2f (+/-%.3f)" % (results['test_fp'].mean(),results['test_fp'].std())
            print classifier.__class__.__name__+" average FN: %.2f (+/-%.3f)" % (results['test_fn'].mean(),results['test_fn'].std())
            
            #Display the mean of the values from the classification report
            scoring = {'accuracy': 'accuracy', 'auc': 'roc_auc', 'precision': 'precision_macro', 'recall' : 'recall_macro', 'f1score' : 'f1_macro'}
            classifier2 = clfs[x]
            results = cross_validate(classifier2, X, y, cv= 10, scoring=list(scoring.values()), 
                         return_train_score=False)
            for sc in range(len(scoring)):
                print(classifier.__class__.__name__+" average %s: %.4f (+/-%.4f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                if list(scoring.values())[sc]=='neg_log_loss' 
                else results['test_%s' % list(scoring.values())[sc]].mean(), 
                results['test_%s' % list(scoring.values())[sc]].std()))
    
         
def main():
    #Functions to call the classification or the feature selection method
    classification(models=['RF'], doKfold= True)   
    #FeatureSelection()
        
main()   
    














        

