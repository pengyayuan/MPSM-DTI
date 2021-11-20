#!usr/bin/env python
#-*- coding: utf-8 -*
#2019-11-27-by-Yayuan Peng, updated in 2020-07-28 by Yayuan Peng;

import os
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, make_scorer
from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
import pandas as pd
import pickle

#reload(sys)
#sys.setdefaultencoding('utf8')
	
# Baggaing Classifier
def bagging_classifier(standardscaler):
	from sklearn.ensemble import BaggingClassifier
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('bagging',BaggingClassifier(random_state=111))])
	param_grid={'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[5],'bagging__n_estimators': range(200,220,3)}
	return pipe, param_grid

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(standardscaler):
	from sklearn.ensemble import GradientBoostingClassifier
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('GBDT',GradientBoostingClassifier(random_state=111))])
	param_grid = {'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[5],'GBDT__learning_rate' : [ 0.4, 0.5, 0.6], 'GBDT__n_estimators' : range(1400,1800,25)}
	return pipe, param_grid

# Random Forest Classifier
def random_forest_classifier(standardscaler):
	from sklearn.ensemble import RandomForestClassifier
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('RF',RandomForestClassifier(random_state=111))])	
	param_grid={'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[5],'RF__n_estimators':np.arange(200,800,50),'RF__max_features':['auto']}
	return pipe, param_grid
	
# KNN Classifier
def knn_classifier(standardscaler):
	from sklearn.neighbors import KNeighborsClassifier
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('knn',KNeighborsClassifier())])
	param_grid={'sample__k_neighbors':[5],'knn__n_neighbors':[3,5,7], 'knn__weights':['uniform']}
	return pipe, param_grid

# Decision Tree Classifier
def decision_tree_classifier(standardscaler):
	from sklearn import tree
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('DT',tree.DecisionTreeClassifier(random_state=111))])
	param_grid={'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[3,5],'DT__criterion':['gini','entropy']}
	return pipe, param_grid

# SVM Classifier
def svm_classifier(standardscaler):
	from sklearn.svm import SVC
	sampling=SMOTE(random_state=111)
	pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('svm',SVC(kernel='rbf', random_state=111))])
	param_grid = {'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[5],'svm__C': [80, 100,120,140,160,180,5000]}
	return pipe, param_grid

# Adaboost Classifier
# def adaboost_classifier(standardscaler):
	# from sklearn.ensemble import AdaBoostClassifier
	# sampling=SMOTE(random_state=111)
	# pipe=Pipeline_imb([('sample',sampling),('sacler',standardscaler),('adboost',AdaBoostClassifier(random_state=111))])
	# param_grid = {'sacler__sparse_threshold':[0.3],'sample__k_neighbors':[3],'adboost__learning_rate' : [0.5], 'adboost__n_estimators' : range(490,500,10)}
	# return pipe, param_grid
  		
def read_data(data_file):
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.compose import ColumnTransformer
	data = pd.read_csv(data_file,sep="\t")
	#drop nan inf
	data=data[~data.isin([np.nan,np.inf,-np.inf]).any(1)].dropna()
	X,y = data.loc[:,'Morgan1':'pssm_composition399'], data.loc[:,['class']]
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 111)
	#ColumnTransformer
	pssm=[]
	for i in range(0,400):
		pssm.append('pssm_composition'+str(i))	
	standardscaler = ColumnTransformer(transformers=[('standardscaler', StandardScaler(),pssm)], remainder='passthrough')
	y_train=np.ravel(y_train)
	y_test=np.ravel(y_test)
	return X_train,X_test,y_train,y_test,standardscaler
	
if __name__ == '__main__':
	kfold=KFold(n_splits=10,shuffle=True,random_state=111)
	data_file = "DTI-feature-Morgan.txt" 
	test_classifiers = ['RF1']
	classifiers = {'bagging':bagging_classifier,'GBDT':gradient_boosting_classifier,'DT':decision_tree_classifier,'RF1':random_forest_classifier,'SVM':svm_classifier,'KNN':knn_classifier}
	X_train, X_test, y_train,y_test,standardscaler= read_data(data_file)
	
	scoring={'auc':make_scorer(roc_auc_score),
			'accuracy': make_scorer(accuracy_score),
			'bacc': make_scorer(balanced_accuracy_score),
			'f1': make_scorer(f1_score),
			'npv': make_scorer(precision_score,pos_label=0),
			'ppv': make_scorer(precision_score,pos_label=1),
			'sp': make_scorer(recall_score,pos_label=0),
			'se': make_scorer(recall_score,pos_label=1)}			
	
	for classifier in test_classifiers:
		pipe, param_grid= classifiers[classifier](standardscaler)
		grid=GridSearchCV(pipe,param_grid=param_grid,cv=kfold,n_jobs=-1,scoring=scoring,refit='se')
		model=grid.fit(X_train, y_train)
		pgs=grid.best_params_
		with  open("Morgan_parameters.txt","a") as f_p:
			f_p.write('The best parameters are %s' %(pgs))
			f_p.write("\n")
		model=grid.best_estimator_
		with open('%s_Morgan.pickle' %classifier,'wb') as f0:
			pickle.dump(model,f0)
			
		scores=grid.cv_results_
		gcr=pd.DataFrame(scores)
		gcr.to_csv("Morgan_%s_grid_search.csv" %classifier)
		rank=scores['rank_test_se'].tolist()
		index=rank.index(1)
		
		mean_auc=scores['mean_test_auc'][index]
		std_auc=scores['std_test_auc'][index]
		mean_acc=scores['mean_test_accuracy'][index]
		std_acc=scores['std_test_accuracy'][index]
		mean_bacc=scores['mean_test_bacc'][index]
		std_bacc=scores['std_test_bacc'][index]
		mean_f1=scores['mean_test_f1'][index]
		std_f1=scores['std_test_f1'][index]
		mean_npv=scores['mean_test_npv'][index]
		std_npv=scores['std_test_npv'][index]
		mean_ppv=scores['mean_test_ppv'][index]
		std_ppv=scores['std_test_ppv'][index]
		mean_sp=scores['mean_test_sp'][index]
		std_sp=scores['std_test_sp'][index]
		mean_se=scores['mean_test_se'][index]
		std_se=scores['std_test_se'][index]
		with open("Morgan_train_results.txt","a") as f_train:
			if os.path.getsize("Morgan_train_results.txt")==0:
				f_train.write(('\t').join(('model','AUC','F1','ACC','Balanced_ACC','NPV','PPV','SP','SE'))+'\n')
				f_train.write('Morgan-%s' %classifier)
				f_train.write('\t')
				f_train.write('%.2f''+/-''%.2f''\t' % (100*mean_auc,100*std_auc))
				f_train.write('%.2f''+/-''%.2f''\t' % (100*mean_f1,100*std_f1))
				f_train.write('%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t' % (100*mean_acc,100*std_acc, 100*mean_bacc, 100*std_bacc))
				f_train.write('%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f' %(100*mean_npv,100*std_npv,100*mean_ppv,100*std_ppv,100*mean_sp,100*std_sp,100*mean_se,100*std_se))
				f_train.write("\n")
			else:
				f_train.write('Morgan-%s' %classifier)
				f_train.write('\t')
				f_train.write('%.2f''+/-''%.2f''\t' % (100*mean_auc,100*std_auc))
				f_train.write('%.2f''+/-''%.2f''\t' % (100*mean_f1,100*std_f1))
				f_train.write('%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t' % (100*mean_acc,100*std_acc, 100*mean_bacc, 100*std_bacc))
				f_train.write('%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f''\t''%.2f''+/-''%.2f' %(100*mean_npv,100*std_npv,100*mean_ppv,100*std_ppv,100*mean_sp,100*std_sp,100*mean_se,100*std_se))
				f_train.write("\n")
		
		predict = model.predict(X_test)
		roc_auc = metrics.roc_auc_score(y_test, predict)
		acc = metrics.accuracy_score(y_test, predict)
		bacc = metrics.balanced_accuracy_score(y_test, predict)
		f1 = metrics.f1_score(y_test, predict)
		npv = metrics.precision_score(y_test, predict,pos_label=0)
		ppv = metrics.precision_score(y_test, predict,pos_label=1)
		sp = metrics.recall_score(y_test, predict,pos_label=0)
		se = metrics.recall_score(y_test, predict,pos_label=1)
		with open("Morgan_test_results.txt","a") as f_test:
			if os.path.getsize("Morgan_test_results.txt")==0:
				f_test.write(('\t').join(('model','AUC','F1','ACC','Balanced_ACC','NPV','PPV','SP','SE'))+'\n')
				f_test.write('Morgan-%s' %classifier)
				f_test.write('\t')
				f_test.write('%.2f''\t'% (100 * roc_auc))
				f_test.write('%.2f''\t' % (100 * f1))
				f_test.write('%.2f''\t''%.2f''\t' % (100 * acc, 100 * bacc))
				f_test.write('%.2f''\t''%.2f''\t''%.2f''\t''%.2f' % (100*npv, 100 * ppv, 100 * sp, 100*se))
				f_test.write("\n")
			else:
				f_test.write('Morgan-%s' %classifier)
				f_test.write('\t')
				f_test.write('%.2f''\t'% (100 * roc_auc))
				f_test.write('%.2f''\t' % (100 * f1))
				f_test.write('%.2f''\t''%.2f''\t' % (100 * acc, 100 * bacc))
				f_test.write('%.2f''\t''%.2f''\t''%.2f''\t''%.2f' % (100*npv, 100 * ppv, 100 * sp, 100*se))
				f_test.write("\n")
