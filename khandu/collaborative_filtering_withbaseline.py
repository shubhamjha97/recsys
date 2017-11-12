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
#import time
from copy import deepcopy


def spearman(pred, truth):
    n = 
    return 1-((6*(np.sum(np.square(pred-truth)/pred.shape[0])))/((n**3)-n))

def RMSE(pred, truth):
    return np.sqrt(np.sum(np.square(pred-truth)/pred.shape[0]))


train=recsys_utils.read_train()
test=recsys_utils.read_test_table()
truth=test['rating'].as_matrix()
user_map=recsys_utils.read_user_map()
movie_map=recsys_utils.read_movie_map()

def baseline_estimator(mat,mode,usr_param,movie_param,overall_mean_movie_rating):
    if mode=='user':
        #sum_usr = mat.sum(axis=0)
        #sum_movie = mat.sum(axis=1)
        print 'Mat shape', mat.shape
        return 0
        #nozero_usr = mat.count_nonzero()#np.count_nonzero(mat.todense(),axis=1)
        #nonzero_movie = np.count_nonzero(mat,axis=0)
        #mean_user_rating = sum_usr[usr_param]/nozero_usr[usr_param]
        #mean_movie_rating = sum_movie[movie_param]/nonzero_movie[movie_param]
        return (mean_user_rating-overall_mean_movie_rating)+(mean_movie_rating-overall_mean_movie_rating)
    
def baseline_Wrt_user(mat,overall_mean_movie_rating):
    user_sum_mat = mat.sum(axis = 1)
    nozero_usr = np.count_nonzero(mat,axis = 1)
    movie_sum_mat = mat.sum(axis = 0)
    nonzero_movie = np.count_nonzero(mat,axis = 0)
    print user_sum_mat
    

def predict_baseline(mat, dist_mat, test, user_map, movie_map, n,mode,temp2,usr_mean,movie_mean):
    
    pred=[]
    overall_mean_movie_rating = mat.sum()/mat.count_nonzero()
    print overall_mean_movie_rating
    
    #mean_rating_user = baseline_Wrt_user(mat,overall_mean_movie_rating)
   
    test = test.head(100)
    if mode=='user':
        for idx,row in test.iterrows():
            dist=np.reshape(dist_mat[:, user_map[row['userId']]], [len(dist_mat),1])
            
            usr_ratings=temp2[:, movie_map[row['movieId']]].todense()
            
            temp_rating_dist=zip(dist.tolist(), usr_ratings.tolist())
            temp_rating_dist.sort(reverse=True)
            temp_rating_dist=temp_rating_dist[1:]
            rating = usr_mean[user_map[row['userId']]] + movie_mean[0,movie_map[row['movieId']]] - overall_mean_movie_rating
            print " movie rating ::  " , movie_mean[0,movie_map[row['movieId']]] 
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
                print "User rating:: " , usr_mean[user_map[row['userId']]]
               
                print (rating-similar_rating/den)
                print similar_rating/den," :: ",similar_rating, " den:: ",den
                print "Negative value encountered"
                #time.sleep(5)  
            pred.append(rating)
            print 'Truth:', row['rating'], 'Pred:', rating
    else:
        pass
    
    return np.array(pred)
                
			
            
		# iterate over test cases
'''		for idx,row in test.iterrows(): #########remove head
			dist=np.reshape(dist_mat[:, user_map[row['userId']]], [len(dist_mat),1])
			usr_ratings=mat[:, movie_map[row['movieId']]].todense()

			temp_rating_dist=zip(dist.tolist(), usr_ratings.tolist())
			temp_rating_dist.sort(reverse=True)
			temp_rating_dist=temp_rating_dist[1:]
          rating = overall_mean_movie_rating + basline_estimator(mat,mode='user',user_map[row['userId']],movie_map[row['movieId']],overall_mean_movie_rating)
            
          similar_rating = 0
			c=1
			den=0
			for i in range(len(temp_rating_dist)):
				if c>=n:
					break
				elif temp_rating_dist[i][1][0]!=0:
					#print 'Dist:', temp_rating_dist[i][0][0], 'Rating:', temp_rating_dist[i][1][0]
					similar_rating+=(temp_rating_dist[i][1][0]-baseline_esitimator(mat,mode='user',user_map[row['userId']],))*temp_rating_dist[i][0][0]
					den+=temp_rating_dist[i][0][0]
				c+=1
			if den==0:
				den=1
			rating+=similar_rating/den
			pred.append(rating)
			print 'Truth:', row['rating'], 'Pred:', rating

	else: # add item-item
		pass
	return np.array(pred)'''
    



# User-user collaborative filtering
print ('User-user collaborative filtering....')
print type(train)

counts=Counter(train.nonzero()[0])

count_movie =  Counter(train.nonzero()[1])

means_mat=train.sum(axis=1)
movie_mat=train.sum(axis=0)

print "means_mat_Shape:  ",means_mat.shape[0]
print "Movie _mat_Shape:  ",movie_mat.shape[1]

for i in range(means_mat.shape[0]):
		if i in counts.keys():
				means_mat[i,0]=means_mat[i, 0]/counts[i]
		else:
				means_mat[i,0]=0
for i in range(movie_mat.shape[1]):
		if i in count_movie.keys():
				movie_mat[0,i]=movie_mat[0, i]/count_movie[i]
		else:
				movie_mat[0,i]=0



temp=deepcopy(train)
temp2=deepcopy(train)

mask= temp!=0
nonzero_vals=np.array(np.nonzero(temp))
nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])

temp_start_time=time()
for val in nonzero_vals:
    temp[val[0], val[1]]-=means_mat[val[0]]
    temp2[val[0],val[1]] = temp2[val[0],val[1]] - means_mat[val[0]] - movie_mat[0,val[1]] 


print ('Time taken:', time()-temp_start_time)

user_dist = 1-pairwise_distances(temp, metric='cosine')
    
predictions=predict_baseline(train, user_dist, test, user_map, movie_map, 10,'user',temp2,means_mat,movie_mat)
truths=test['rating'].as_matrix()
print RMSE(predictions,truths)

