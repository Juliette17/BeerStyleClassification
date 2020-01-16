import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from joblib import dump, load



def prepare_dataset(beers_df, has_labels = True):
    beers_df = beers_df.drop(['UserId'], axis=1)
    #print (beers_df.describe(include = 'O'))
    #print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
    if has_labels:
        beers_df_y = beers_df['Style'].values
        beers_df = beers_df.drop(['Style'], axis=1)
        beers_df = beers_df.drop(['Name'], axis=1)
    else:
        beers_df = beers_df.drop(['Id'], axis=1)
        beers_df_y = []
    
    boil_mean = np.mean(beers_df['BoilGravity'])
    beers_df['BoilGravity'] = beers_df[['BoilGravity']].fillna(boil_mean)

    pitch_mean = np.mean(beers_df['PitchRate'])
    beers_df['PitchRate'] = beers_df[['PitchRate']].fillna(pitch_mean)

    beers_df_x = beers_df.values
    return beers_df_x, beers_df_y

##################################################################################################Naive Bayes
def naive_bayes(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    accuracy = (y_test == y_pred).sum()/X_test.shape[0]
    print("Accuracy of Gaussian Naive Bayes : ", accuracy)
########################################################decision tree

def decision_tree(X_train, y_train, X_test, y_test):
    min_samples = 200
    for min_samples in range(50, 60, 20):
        print("Min dsmples split: ", min_samples)
        for max_leaf_nodes in range(100, 101, 1):
            print("Max leaf nodes: ", max_leaf_nodes)
            dtc = DecisionTreeClassifier(random_state=17, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples)
            y_pred = dtc.fit(X_train, y_train).predict(X_test)
            accuracy = (y_test == y_pred).sum()/X_test.shape[0]
            print("Accuracy of DecisionTreeClassifier : ", accuracy)
            print("")

#prediction_on_validation_test(90, 16, 52)
#prediction_on_validation_test(90, 16, 68)  0.6962686567164178
def prediction_on_validation_test(n, max_depth, min_samples):
    #preapring training set
    beers_df = pd.read_csv('data/beers.csv')
    print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
    beers_df_x, beers_df_y = prepare_dataset(beers_df)

    #preparing validation set
    beers_validation_df = pd.read_csv('data/beers_test_nostyle.csv')
    beers_vdf_x, beers_vdf_y = prepare_dataset(beers_validation_df, has_labels=False)

    X_train = beers_df_x
    y_train = beers_df_y
    rfc = RandomForestClassifier(n_estimators= n, max_depth = max_depth, min_samples_split=min_samples)
    fitted = rfc.fit(X_train, y_train)
    y_val = fitted.predict(beers_vdf_x)

    lbl_df = pd.DataFrame({'Style': y_val})
    lbl_df.to_csv('data/labels.csv')

def predict_on_valid_set(filename = 'noname'):
    if filename == 'noname':
        prediction_on_validation_test(90, 16, 68)
    else:
        beers_df = pd.read_csv('data/beers.csv')
        print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
        beers_df_x, beers_df_y = prepare_dataset(beers_df)

        #preparing validation set
        beers_validation_df = pd.read_csv('data/beers_test_nostyle.csv')
        beers_vdf_x, beers_vdf_y = prepare_dataset(beers_validation_df, has_labels=False)

        X_train = beers_df_x
        y_train = beers_df_y
        rfc = load(filename)
        fitted = rfc.fit(X_train, y_train)
        y_val = fitted.predict(beers_vdf_x)

        lbl_df = pd.DataFrame({'Style': y_val})
        lbl_df.to_csv('data/labels.csv')

#predict_on_valid_set('rf90_16_70_0.6995102611940298.joblib')
#######################################################################################################random forest
def test_random_forest():
    #preapring training set
    beers_df = pd.read_csv('data/beers.csv')
    print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
    beers_df_x, beers_df_y = prepare_dataset(beers_df)

    results = {}
    times = 20
    for min_samples in range(70, 72, 2):
        for max_depth in range(16, 17, 1):
            accuracy_sum = 0
            for n in range(90, 100, 10):
                accuracy_sum = 0
                for i in range(times):
                    X_train, X_test, y_train, y_test = train_test_split(beers_df_x, beers_df_y, test_size = 0.2)
                    #rfc = RandomForestClassifier(n_estimators= n, max_depth = max_depth, min_samples_split=min_samples)
                    rfc = load('rf90_16_70_0.6995102611940298.joblib')
                    fitted = rfc.fit(X_train, y_train)
                    y_pred = fitted.predict(X_test)
                    
                    accuracy = (y_test == y_pred).sum()/X_test.shape[0]
                    print(accuracy)
                    #dump(rfc, 'rf{}_{}_{}_{}.joblib'.format(n, max_depth, min_samples, accuracy)) 
                     
                    accuracy_sum +=accuracy
                    #print(rfc.decision_path(X_test[1]))
                results[(min_samples, max_depth, n)] = accuracy_sum/times
                print(confusion_matrix(y_test, y_pred))


    for x in results:
        print (x, results[x])

    print(confusion_matrix(y_test, y_pred))

#test_random_forest()
#print(confusion_matrix(y_test, y_pred))


#Linear SVC
def test_svc():
    #accuracy_sum = 0
    beers_df = pd.read_csv('data/beers.csv')
    print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
    beers_df_x, beers_df_y = prepare_dataset(beers_df)

    X_train, X_test, y_train, y_test = train_test_split(beers_df_x, beers_df_y, test_size = 0.2)

    svc = SVC(gamma='scale')
    fitted = svc.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    accuracy = (y_test == y_pred).sum()/X_test.shape[0]
    print(accuracy)
    #accuracy_sum +=accuracy
    #results[(min_samples, max_depth, n)] = accuracy

#test_svc()

#sgd
def test_sgd():
    #accuracy_sum = 0
    beers_df = pd.read_csv('data/beers.csv')
    print(beers_df[['Style']].groupby(['Style'], as_index = False).sum())
    beers_df_x, beers_df_y = prepare_dataset(beers_df)

    X_train, X_test, y_train, y_test = train_test_split(beers_df_x, beers_df_y, test_size = 0.2)

    sgd = SGDClassifier()
    fitted = sgd.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    accuracy = (y_test == y_pred).sum()/X_test.shape[0]
    print(accuracy)
    #accuracy_sum +=accuracy
    #results[(min_samples, max_depth, n)] = accuracy

#test_sgd()

def check_differences(filename1, filename2):
    labels1 = pd.read_csv(filename1)
    labels2 = pd.read_csv(filename2)
    print(sum(labels1['Style'] != labels2['Style']))

check_differences('data/labels1.csv', 'data/labels.csv')