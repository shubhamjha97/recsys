import pandas as pd
import cPickle as pickle
import numpy as np
import os
import scipy
from scipy.sparse import load_npz

def read_train(sparse=False):
	if sparse:
		return scipy.sparse.load_npz('temp_data/train.npz')
	else:
		return scipy.sparse.load_npz('temp_data/train.npz').todense()#.astype(int)

def read_test_table():
	return pickle.load(open('temp_data/test_table.pkl', 'r+'))

def read_movie_map():
	return pickle.load(open('temp_data/movie_map.pkl', 'r+'))

def read_user_map():
	return pickle.load(open('temp_data/user_map.pkl', 'r+'))

if __name__=='__main__':
	train=read_train()
	print train.shape