#!usr/bin/env python
#-*- coding: utf-8 -*
#2019-11-27-by-Yayuan Peng,updated in 2020-08-18

import os
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, make_scorer
import numpy as np
import pandas as pd
import pickle
 
	
def read_data(data_file):
	from sklearn.preprocessing import StandardScaler
	#from sklearn.model_selection import train_test_split
	data = pd.read_csv(data_file,sep="\t")
	#drop nan inf
	data=data[~data.isin([np.nan,np.inf,-np.inf]).any(1)].dropna()
	X_validation = data.loc[:,'Morgan1':'pssm_composition399']
	compound_target_id = data.loc[:,'compound_id':'target_id']
	return X_validation, compound_target_id
    
if __name__ == '__main__':
	for  compound_id in range (1,101):
		data_file = "C-%s_target_feature.txt" %compound_id 
		X_validation, compound_target_id = read_data(data_file)
		with open('SVM_Morgan.pickle','rb') as f_0:
			model=pickle.load(f_0)
			predict_prob = model.predict_proba(X_validation)
			prob_1 = predict_prob[:,1]
			compound_target_id.insert(2,'prob_1', prob_1)
			compound_target_id.to_csv("%s_predict_prob_results.csv" %compound_id, index_label=False)


			
			