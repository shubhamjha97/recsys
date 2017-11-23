# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 07:41:44 2017

@author: Saurabh
"""

import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from time import time
from copy import deepcopy
import evaluation


train=recsys_utils.read_train(sparse = True)
test=recsys_utils.read_test_table()
truth=test['rating'].as_matrix()
user_map=recsys_utils.read_user_map()
movie_map=recsys_utils.read_movie_map()


def subtract_mean(mat_,type='user'):
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
   
    mask= mat!=0
    nonzero_vals=np.array(np.nonzero(mat))
    nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])
    for val in nonzero_vals:
        mat[val[0], val[1]]-=means_mat[val[0]]
    return mat

    

def predict_baseline(mat, dist_mat, test, user_map, movie_map, n,mode,temp2,usr_mean,movie_mean):
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
    temp2 (2D numpy array): Input Matrix modified with BaseLine approach 
    mode ['user', 'item']
    usr_mean(1D numpy array) : Array of Users Mean 
    movie_mean(1D numpy array)  :Array of Movie Mean.
    
    Returns:
    pred (1D numpy array): array containing predictions to the test data.
    '''

    
    pred=[]
    print "Entered Prediction Function"
    overall_mean_movie_rating = mat.sum()/mat.count_nonzero()
    print "Overall Mean Movie Rating ",overall_mean_movie_rating
    no_of_ratings = 0
    no_of_zero = 0
    
   
    test = test.head(10000)
    if mode=='user':
        for idx,row in test.iterrows():
            dist=np.reshape(dist_mat[:, user_map[row['userId']]], [len(dist_mat),1])
            
            usr_ratings=temp2[:, movie_map[row['movieId']]].todense()
            
            temp_rating_dist=zip(dist.tolist(), usr_ratings.tolist())
            temp_rating_dist.sort(reverse=True)
            temp_rating_dist=temp_rating_dist[1:]
            rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
            similar_rating = 0
            c = 1
            den = 0
            for i  in range(len(temp_rating_dist)):
                if c>=n:
                    break
                elif temp_rating_dist[i][1][0]!=0:
                    similar_rating+=(temp_rating_dist[i][1][0]+overall_mean_movie_rating)*temp_rating_dist[i][0][0]
                    den+=temp_rating_dist[i][0][0]
                c+=1
                
            if den==0:
                den=1
            rating+=similar_rating/den
            if rating>5:
                rating=5
            if rating<0:
                rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
            pred.append(rating)
    else:
        print temp2.shape
        for idx,row in test.iterrows(): 
            dist=np.reshape(dist_mat[:, movie_map[row['movieId']]], [len(dist_mat),1])
            movie_ratings=temp2[:, user_map[row['userId']]].todense()
            temp_rating_dist=zip(dist.tolist(), movie_ratings.tolist())
            temp_rating_dist.sort(reverse=True)
            temp_rating_dist=temp_rating_dist[1:]
            no_of_ratings+=1
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
            
            if rating<=0:
                rating = usr_mean[user_map[row['userId']]] + movie_mean[movie_map[row['movieId']]] - overall_mean_movie_rating
                no_of_zero+=1
            pred.append(rating)
            #print "Predicted::",rating,"  Actual:: ",row['rating']
    return np.array(pred)
                


# User-user collaborative filtering
print ('User-user collaborative filtering....')
print type(train)

counts=Counter(train.nonzero()[0])

count_movie =  Counter(train.nonzero()[1])

means_mat=np.squeeze(np.sum(np.array(train.todense()), axis=1))
movie_mat=np.squeeze(np.sum(np.array(train.todense()), axis=0))

print "means_mat_Shape:  ",means_mat.shape
print "Movie _mat_Shape:  ",movie_mat.shape

for i in range(means_mat.shape[0]):
		if i in counts.keys():
				means_mat[i]=means_mat[i]/counts[i]
		else:
				means_mat[i]=0
for i in range(movie_mat.shape[0]):
		if i in count_movie.keys():
				movie_mat[i]=movie_mat[i]/count_movie[i]
		else:
				movie_mat[i]=0



temp=deepcopy(train)
temp2=deepcopy(train)

mask= temp!=0
nonzero_vals=np.array(np.nonzero(temp))
nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])

temp_start_time=time()
print len(nonzero_vals)
for val in nonzero_vals:
    temp2[val[0],val[1]] = temp2[val[0],val[1]] - means_mat[val[0]] - movie_mat[val[1]]

print 'means'
means_mat=np.squeeze(means_mat)
movie_mat=np.squeeze(movie_mat)
print means_mat.shape
print movie_mat.shape

print ('Time taken:', time()-temp_start_time)

user_dist = 1-pairwise_distances(subtract_mean(temp), metric='cosine')
start_time_item = time()
predictions_usr=predict_baseline(train, user_dist, test, user_map, movie_map, 10,'user',temp2,means_mat,movie_mat)
print 'User-User-> Total time:', time()- start_time_item
predictions_usr=np.squeeze(predictions_usr)
print 'User-User-> Total time:', time()- start_time_item
print 'User-User-> RMSE:', evaluation.RMSE(predictions_usr, truth[0:10000])
print 'spearman_rank_correlation', evaluation.spearman_rank_correlation(predictions_usr, truth[0:10000])
print 'Precision on top K' , evaluation.top_k_precision(predictions_usr, test.head(10000), means_mat, user_map)


print 'Item-item collaborative filtering....'
start_time_item=time()
item_dist=1-pairwise_distances(subtract_mean(train.T), metric='cosine')
print 'Time taken to calculate distances:', time()-start_time_item
temp2 = temp2.T
predictions_mov=predict_baseline(train.T, item_dist, test, user_map, movie_map, 10,'item',temp2,means_mat,movie_mat)
predictions=np.squeeze(predictions_mov)
print 'Item-item-> Total time:', time()- start_time_item
print 'Item-item-> RMSE:', evaluation.RMSE(predictions, truth[0:10000])
print 'spearman_rank_correlation', evaluation.spearman_rank_correlation(predictions, truth[0:10000])
print 'Precision on top K' , evaluation.top_k_precision(predictions, test.head(10000), movie_mat, movie_map, 5, False)



