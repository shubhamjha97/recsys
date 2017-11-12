import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from time import time
from copy import deepcopy
import evaluation
from tqdm import tqdm

def subtract_mean(mat_):#,type='user'):
	'''
	Subtract row means from matrix mat_.
	Inputs:
	mat_ (2D numpy array): matrix from which the mean needs to be subtracted.
	
	Returns:
	mat (2D numpy array): Matrix with row means subtracted.
	'''
	mat=deepcopy(mat_)
	counts=Counter(mat.nonzero()[0])
	means_mat=mat.sum(axis=1)
	means_mat=np.reshape(means_mat, [means_mat.shape[0], 1])
	for i in range(means_mat.shape[0]):
				if i in counts.keys():
					means_mat[i,0]=means_mat[i, 0]/float(counts[i])
				else:
					means_mat[i,0]=0
	# Subtract means from non zero values in the matrix
	mask= mat!=0
	nonzero_vals=np.array(np.nonzero(mat))
	nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])
	for val in nonzero_vals:
		mat[val[0], val[1]]-=means_mat[val[0]]
	return mat

def predict(mat, dist_mat, test, user_map, movie_map, n=10, mode='user'):
	'''
	Function to predict the ratings by users on movies in the test dataframe.
	This function implements both user-user and item-ite collaborative filtering.

	Inputs:
	mat (2D numpy array): input train matrix
	dist_mat (2D numpy array): matrix where the [i, j]th element is the cosine similarity between the ith and jth item/user.
	test(pandas dataframe): pandas test dataframe
	user_map (python dict): user mappings
	movie_map (python dict): movie mappings
	n (int): Number of most similar users/items to consider for prediction
	mode ['user', 'item']

	Returns:
	pred (1D numpy array): array containing predictions to the test data.
	'''
	pred=[]
	if mode=='user':
		# iterate over test cases
		for idx,row in test.iterrows():
			dist=np.reshape(dist_mat[:, user_map[row['userId']]], [len(dist_mat),1])
			usr_ratings=mat[:, movie_map[row['movieId']]].todense()
			temp_rating_dist=zip(dist.tolist(), usr_ratings.tolist())
			temp_rating_dist.sort(reverse=True)
			temp_rating_dist=temp_rating_dist[1:]
			rating=0
			c=1
			den=0
			for i in range(len(temp_rating_dist)):
				if c>=n:
					break
				elif temp_rating_dist[i][1][0]!=0:
					rating+=temp_rating_dist[i][1][0]*temp_rating_dist[i][0][0]
					den+=temp_rating_dist[i][0][0]
				c+=1
			if den==0:
				den=1
			rating=rating/den
			pred.append(rating)


	else:
		for idx,row in test.iterrows():
			dist=np.reshape(dist_mat[:, movie_map[row['movieId']]], [len(dist_mat),1])
			movie_ratings=mat[:, user_map[row['userId']]].todense()
			temp_rating_dist=zip(dist.tolist(), movie_ratings.tolist())
			temp_rating_dist.sort(reverse=True)
			temp_rating_dist=temp_rating_dist[1:]
			rating=0
			c=1
			den=0
			for i in range(len(temp_rating_dist)):
				if c>=n:
					break
				elif temp_rating_dist[i][1][0]!=0:
					rating+=temp_rating_dist[i][1][0]*temp_rating_dist[i][0][0]
					den+=temp_rating_dist[i][0][0]
				c+=1
			if den==0:
				den=1
			rating=rating/den
			pred.append(rating)
	return np.array(pred)

if __name__=='__main__':

	# Read files	
	train=recsys_utils.read_train(sparse=True)
	test=recsys_utils.read_test_table().head(10000)
	truth=test['rating'].head(10000).as_matrix()
	user_map=recsys_utils.read_user_map()
	movie_map=recsys_utils.read_movie_map()

	# User-user collaborative filtering
	# user_means=np.squeeze(np.sum(np.array(train.todense()), axis=1))
	user_means=np.squeeze(np.sum(np.array(train.todense()), axis=1))
	user_means=np.divide(user_means, (np.array(train.todense())!=0).sum(1))
	print 'User-user collaborative filtering....'
	start_time_user=time()
	user_dist=1-pairwise_distances(subtract_mean(train.astype('float32')), metric='cosine')
	print 'Time taken to calculate distances:', time()-start_time_user
	predictions=predict(train, user_dist, test, user_map, movie_map, 10)
	print 'User-user-> Total time:', time()- start_time_user
	print 'User-user-> RMSE:', evaluation.RMSE(predictions, truth)
	print 'spearman_rank_correlation', evaluation.spearman_rank_correlation(predictions, truth)
	print 'top k precision:', evaluation.top_k_precision(predictions, test, user_means, user_map, k=5)
	print 'Total time:', time()-start_time_user

	# Item-item collaborative filtering
	# item_means=np.squeeze(np.sum(np.array(train.T.todense()), axis=1))
	item_means=np.squeeze(np.sum(np.array(train.T.todense()), axis=1))
	item_means=np.divide(user_means, (np.array(train.T.todense())!=0).sum(1))
	print 'Item-item collaborative filtering....'
	start_time_item=time()
	item_dist=1-pairwise_distances(subtract_mean(train.T.astype('float32')), metric='cosine')
	print 'Time taken to calculate distances:', time()-start_time_item
	predictions=predict(train.T, item_dist, test, user_map, movie_map, 10, 'item')
	print 'Item-item-> Total time:', time()- start_time_item
	print 'Item-item-> RMSE:', evaluation.RMSE(predictions, truth)
	print 'spearman_rank_correlation', evaluation.spearman_rank_correlation(predictions, truth)
	print 'top k precision:', evaluation.top_k_precision(predictions, test, item_means, movie_map, k=5, user_=False)
	print 'Total time:', time()-start_time_item