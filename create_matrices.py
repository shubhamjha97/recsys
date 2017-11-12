import pandas as pd
import numpy as np
import cPickle as pickle
from scipy.sparse import csr_matrix
import scipy
import os
from time import time
start_time=time()

TEST_SIZE=0.2
SPARSE=True #do not change
SHUFFLE=True

# Read file from disk
path='data/ratings.dat'
print 'Data path:', path
table=pd.read_table(path, sep='::', header=None, 
		names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

no_entries=table.shape[0]

# Shuffle data
if SHUFFLE:
	table = table.sample(frac=1).reset_index(drop=True)

# Find total no. of users and movies
movie_id_list=table['movieId'].unique()
user_id_list=table['userId'].unique()
no_users=len(user_id_list)
no_movies=len(movie_id_list)
print 'Overall dataset: total ratings=', no_entries
print 'Overall dataset: total users=', len(user_id_list)
print 'Overall dataset: total movies=', len(movie_id_list)

# Map movies and users
movie_map={}
for ix, m_id in enumerate(movie_id_list):
	movie_map[m_id]=ix

user_map={}
for ix, m_id in enumerate(user_id_list):
	user_map[m_id]=ix

# Split train and test
train_table=table.head(int((1-TEST_SIZE)*no_entries))
test_table=table.tail(int(TEST_SIZE*no_entries))

print 'Train set: total ratings=', train_table.shape[0]
print 'Train set: total users=', len(train_table['userId'].unique())
print 'Train set: total movies=', len(train_table['movieId'].unique())

print 'Test set: total ratings=', test_table.shape[0]
print 'Test set: total users=', len(test_table['userId'].unique())
print 'Test set: total movies=', len(test_table['movieId'].unique())

# Create Matrices
train = np.zeros([len(user_map), len(movie_map)])
# test=np.zeros([len(user_map), len(movie_map)])

print 'Creating matrices...'
create_start_time=time()
for idx,row in train_table.iterrows():
	train[user_map[row['userId']], movie_map[row['movieId']]]=row['rating']

# for idx,row in test_table.iterrows():
# 	test[user_map[row['userId']], movie_map[row['movieId']]]=row['rating']
print 'Time taken to create matrices:', time()-create_start_time

# Sanity Check
print 'Train:',train.shape#, 'Test:',test.shape
print 'Density:', 100.0*float(np.count_nonzero(train))/(no_users*no_movies), '%'#, 100.0*float(np.count_nonzero(test))/(no_users*no_movies),'%'

# Convert to Compressed Row Sparse
sparse_start_time=time()
if SPARSE: # remove
	train=scipy.sparse.csr_matrix(train)
	# test=scipy.sparse.csr_matrix(test)
print 'Time taken to convert to sparse:', time()-sparse_start_time

# Write Matrices to file
if SPARSE:
	# save npz files
	scipy.sparse.save_npz('temp_data/train.npz', train)
	# scipy.sparse.save_npz('temp_data/test.npz', test)
	if 'train.npy' in os.listdir('temp_data'):
		os.remove('temp_data/train.npy')
	# if 'test.npy' in os.listdir('temp_data'):
	# 	os.remove('temp_data/test.npy')
# else:
# 	# save npy files
# 	np.save('temp_data/train', train)
# 	np.save('temp_data/test', test)
# 	if 'train.npz' in os.listdir('temp_data'):
# 		os.remove('temp_data/train.npz')
# 	if 'test.npz' in os.listdir('temp_data'):
# 		os.remove('temp_data/test.npz')

print 'train.npz saved to disk....'
with open('temp_data/movie_map.pkl', 'w+') as f:
	pickle.dump(movie_map, f)
with open('temp_data/user_map.pkl', 'w+') as f:
	pickle.dump(user_map, f)
with open('temp_data/test_table.pkl', 'w+') as f:
	pickle.dump(test_table, f)
print 'movie_map.pkl and user_map.pkl saved to disk....'
print 'Script runtime:', time()-start_time