import pandas as pd
import numpy as np

from adapter import BaseEstimator
from adapter import CrossValidator

class ClassificationTester(BaseEstimator):
    
    def fit(self, train_X, train_y, params):
        pass
        
    def score(self, valid_X, valid_y):
        zero_prop = np.sum(valid_y==0)/len(valid_y)
        return zero_prop
    
    def predict(self, test_X):
        pass


def test_binary_classification_scheme():

	data = np.arange(1000*3).reshape(1000, 3)
	target = np.asarray([0]*250 + [1]*(750)).reshape(-1, 1)
	data = np.hstack((target, data)).astype(np.float)
	df = pd.DataFrame(data=data, columns=['target', 'f1', 'f2', 'f3'])

	myest = ClassificationTester()
	cv = CrossValidator(df, ['f1', 'f2', 'f3'], 'target', K=5, cross_validation_scheme='binary_classification')

	assert(cv.evaluate_fold(myest, 0, [0, 1, 2]) == 0.25)
	assert(cv.evaluate_fold(myest, 1, [0, 1, 2]) == 0.25)
	assert(cv.evaluate_fold(myest, 2, [0, 1, 2]) == 0.25)
	assert(cv.evaluate_fold(myest, 3, [0, 1, 2]) != 0.25)



