import numpy as np

def RMSE(pred, truth):
	'''
	Calculate Root Mean Square Error (RMSE).

	Inputs:
	pred (1D numpy array): numpy array containing predicted values.
	truth (1D numpy array): numpy array containing the ground truth values.

	Returns:
	rmse (float): The Root Mean Square Error.
	'''
	return np.sqrt(np.sum(np.square(pred-truth)/float(pred.shape[0])))

def RMSE_mat(matA, matB):
	'''
	Calculate Root Mean Square Error (RMSE) between two matrices. Mainly used
	to find error original and reconstructed matrices while working with 
	matrix decompositions.

	Inputs:
	matA (2D numpy array): Matrix A 
	matB (2D numpy array): Matrix B
	
	Returns:
	rmse (float): Root Mean Square Error.
	'''
	return np.sqrt(np.sum(np.square(matA-matB))/(matA.shape[0]*matA.shape[1]))

def top_k_precision(pred, test, means_, map_, k=5, user_=True):
	'''
	Calculate Precision@top k.

	Inputs:
	pred (1D numpy array): numpy array containing predicted values.
	truth (1D numpy array): numpy array containing the ground truth values.

	Returns:
	(float): average Precision@top k.
	'''
	# THRESHOLD=3.5
	# K=5
	K=k
	precision_list=[]
	test['prediction']=pred

	if user_==True:
		# unique_users=test['userId'].unique()
		unique_values=test['userId'].unique()
	else:
		# unique_users=test['movieId'].unique()
		unique_values=test['movieId'].unique()

	for val in unique_values:
		THRESHOLD=means_[map_[val]]
		if user_==True:
			temp_df=test[test['userId']==val].copy(deep=True)
		else:
			temp_df=test[test['movieId']==val].copy(deep=True)
		temp_df.sort_values('prediction', inplace=True, ascending=False)
		temp_df=temp_df.head(K)
		temp_df['rating']=temp_df['rating']>=THRESHOLD
		temp_df['prediction']=temp_df['prediction']>=THRESHOLD
		no_equals = temp_df[temp_df["rating"] == temp_df["prediction"]].shape[0]
		print temp_df.shape
		temp_precision=no_equals/float(temp_df.shape[0])
		# print no_equals, temp_precision
		precision_list.append(temp_precision)
	return np.mean(np.array(precision_list))

def spearman_rank_correlation(pred, truth):
	'''
	Calculate Spearman Rank Correlation.

	Inputs:
	pred (1D numpy array): numpy array containing predicted values.
	truth (1D numpy array): numpy array containing the ground truth values.

	Returns:
	rho (float): Spearman Rank Correlation
	'''
	d=np.sum(np.square(pred-truth))
	n=len(pred)
	rho=1-6.0*d/(n*(n*n-1))
	return rho

if __name__=='__main__':
	shp=[100, 100]
	a=np.random.randint(1, 6, shp)
	b=np.random.randint(1, 6, shp)
	print RMSE_mat(a,b)
	print spearman_rank_correlation(a,b)