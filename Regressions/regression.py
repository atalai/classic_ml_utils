#   @@@@@@@@@@@@@@@@@@@@@@@@
#   **Code by Aron Talai****
#   @@@@@@@@@@@@@@@@@@@@@@@@

# Various regression (linear and non-linear) tools

#def libs
import os
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

class linear_regression:

	def general_regression (self, x, y, kernel_type):
		''' Regression with numpy
				# simple linear regression
				# kernel_type = 1
				# model_coef[0]--> Slope 
				# model_coef[1]--> Intercept

				# kernel_type = 2 2nd order and 3 is cubic
				#model_coef_1 = np.polyfit(x,y, kernel_type)
				if kernel type is not 1, root mean squared will return 0 ''' 
		model_coef = np.polyfit(x,y, kernel_type)
		
		if kernel_type == 1:

			# calculate r**2 value
			predicted_y = model_coef[0]*x + model_coef[1]
			y_residual = y - predicted_y
			SS_residual = sum(pow(y_residual,2))
			SS_total = len(y) * np.var(y)
			root_mean_squared =  1 - SS_residual/SS_total

		else: root_mean_squared = 0
		return ([model_coef,root_mean_squared])

	def sklearn_linear_regression(self, x, y, z):
		'''performs standard linear regression on x and y, and provides 
		1) slope 2)intercept, 3)R**2 values in array form and 4) the perdicted values
		on the z dataset'''
		model = LinearRegression().fit(x, y)
		if (z != 0):

			y_pred = model.predict(z)
			return [[model.coef_,model.intercept_,model.score(x, y)],[y_pred] ]

		else:
			return [[model.coef_,model.intercept_,model.score(x, y)],[0] ]

	def statsmodel_linear_regression(self,x,y):
		'''Comprehensive linear regression implemented in statsmodel'''
		# statsmodel needs the the first coulumn on x to be one
		x = sm.add_constant(x)
		model = sm.OLS(y, x)
		results = model.fit()
		print (results.summary())
	
	def sgd_regression(slef,x,y, prediction_set):
		''' Perfoms SGD regression by taking x, and y and return the fit model. Attributes need to be called on sgd_regression
		coef_, intercept_ ,average_coef_ : array, shape (n_features,) ,average_intercept_, n_iter_ : int'''
		regr = SGDRegressor(max_iter=1000, tol=1e-3)

		# SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
		#        epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
		#        learning_rate='invscaling', loss='squared_loss', max_iter=1000,
		#        n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
		#        shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
		#        warm_start=False)


		model = regr.fit(x, y)  

		if isinstance(prediction_set, np.ndarray):

			y_pred = model.predict(prediction_set)
			return [regr.score(x,y), regr.predict(prediction_set)]

		else: 

			return  [regr.score(x,y) ]

	def rf_regression (self, x, y, prediction_set):
		''' Perfoms random forest regression by taking x, and y and returns the feature_importances_, and r**2 score'''
		regr = RandomForestRegressor(max_depth=50, random_state=1993, n_estimators=50)

		# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
		#            max_features='auto', max_leaf_nodes=None,
		#            min_impurity_decrease=0.0, min_impurity_split=None,
		#            min_samples_leaf=1, min_samples_split=2,
		#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
		#            oob_score=False, random_state=0, verbose=0, warm_start=False)
		
		model = regr.fit(x, y)  

		if isinstance(prediction_set, np.ndarray):

			y_pred = model.predict(prediction_set)
			return [regr.feature_importances_, regr.score(x,y), regr.predict(prediction_set)]

		else: 

			return  [regr.feature_importances_, regr.score(x,y) ]

	
	def regression_plot(self, x, y, model_coef ):
		''' Shows regression plots by taking x, and y and the model coeficients from the
		general_regression function'''
		plt.plot(x,y,'o')
		xp = np.linspace(0,60,60)
		plt.plot(xp,np.polyval(model_coef,xp),'-')
		plt.show()

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
x = np.array([5, 15, 25, 35, 45, 55])

predictive_model = linear_regression()

# #model_coef = predictive_model.general_regression(x,y,1)[0]
# model_coef = predictive_model.sgd_regression(x,y)
# model_vals = [model_coef.coef_ , model_coef.intercept_]
# predictive_model.regression_plot(x,y,model_vals)

print (predictive_model.rf_regression(x,y,x))
print (predictive_model.sgd_regression(x,y,x))

def main():
    predictive_model = linear_regression()
    predictive_model.statsmodel_linear_regression(x,y)

if __name__ == "__main__":
    main()
