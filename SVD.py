import pandas as pd
import numpy as np
from time import time
import recsys_utils
import os
from copy import deepcopy
import evaluation


def energy_calc(vec, percent_energy_retain):
	'''
	Function to calculate energy of eigenvalues and return the number of 
	eigenvalues to use to retain 'percent_energy_retain'% of total energy.
	
	Inputs-
	vec(1D numpy array): Vector of eigenvalues
	percent_energy_retain(int): percentage of energy to retain

	Returns-
	index(int): number of largest eigenvalues to use.
	'''
	if vec.ndim==2:
		vec=np.squeeze(vec)
	elif percent_energy_retain==0:
		return -1
	print vec[0:10]
	total_energy=np.sum(vec)
	required_energy=percent_energy_retain*total_energy/(100.0)
	index=np.argmin(vec.cumsum() <= required_energy)+1
	return index

def SVD(mat, percent_energy_retain=90, save_factorized=False):
	'''
	Function to perform SVD decomposition of a matrix. This function
	also provides functionality to reduce number of eigenvalues to reduce the
	dimensionality of the factor matrices.
	
	Inputs-
	mat(2D numpy array): The matrix to be decomposed
	percent_energy_retain(int): percentage of energy to retain
	save_factorized(bool): If True, the factor matrices will be saved to disk

	Returns-
	U(2D numpy array): U matrix
	V_t(2D numpy array): Transpose of V matrix
	Sigma(2D numpy array): Sigma Matrix
	'''

	# Calculate U
	vals, vecs=np.linalg.eig(np.dot(mat, mat.T))
	vals=np.absolute(np.real(vals))
	if percent_energy_retain==100:
		no_eigenvalues=np.linalg.matrix_rank(np.dot(mat, mat.T))
	else:
		no_eigenvalues=energy_calc(np.sort(vals)[::-1], percent_energy_retain)
	print 'No of eigenvalues retained:', no_eigenvalues
	indices=np.argsort(vals)[::-1][0:no_eigenvalues]
	U=np.real(vecs[:, indices])

	diag_vals=deepcopy(np.reshape(np.sqrt(np.sort(vals)[::-1])[0:no_eigenvalues], [no_eigenvalues]))

	# Calculate sigma
	sigma=np.zeros([no_eigenvalues, no_eigenvalues])
	np.fill_diagonal(sigma, diag_vals)

	#Calculate V
	V=np.zeros([mat.shape[1], no_eigenvalues])
	for i in range(no_eigenvalues):
		scaling_factor=(1/diag_vals[i])
		V[:, i]= scaling_factor*np.reshape(np.dot(mat.T, np.reshape(U[:, i], [U.shape[0], 1])), [mat.shape[1]])
	V_t=V.T

	if save_factorized:
		np.save('temp_data/U', U)
		np.save('temp_data/V_t', V_t)
		np.save('temp_data/sigma', sigma)
		print 'Matrices saved!'

	return U, V_t, sigma

if __name__=='__main__':
	# Read data
	train=np.array(recsys_utils.read_train())
	test=recsys_utils.read_test_table()
	truth=test['rating'].as_matrix()
	user_map=recsys_utils.read_user_map()
	movie_map=recsys_utils.read_movie_map()

	start_time=time()

	# Subtract means from train
	user_means=np.squeeze(np.sum(train, axis=1))
	user_means=np.divide(user_means, (train!=0).sum(1))
	for i in range(train.shape[0]):
		train[i, :][train[i, :]!=0]-=user_means[i]

	# SVD Decomposition and Reconstruction
	U, V_t, sigma=SVD(train, percent_energy_retain=100, save_factorized=True)
	print 'Factorization Time:', time()-start_time
	reconstructed=np.dot(np.dot(U, sigma), V_t)
	print 'RMSE(reconstruction):', evaluation.RMSE_mat(train, reconstructed)

	# Get Predictions
	pred_mat=train+np.reshape(user_means, [len(user_means), 1])
	rows=[user_map[x] for x in test['userId']]
	cols=[movie_map[x] for x in test['movieId']]
	predictions=pred_mat[rows, cols]
	total_time_svd=time()-start_time
	print 'RMSE:', evaluation.RMSE(np.array(predictions), truth)
	print 'spearman_rank_correlation', evaluation.spearman_rank_correlation(np.array(predictions), truth)
	print 'Top k Precision(k=5):', evaluation.top_k_precision(predictions, 
		test, user_means, user_map, 5)
	print 'Total SVD time:', total_time_svd