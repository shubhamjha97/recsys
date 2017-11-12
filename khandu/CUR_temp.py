# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 20:59:26 2017

@author: Saurabh

"""

import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
import time
from copy import deepcopy
from math import sqrt
from scipy.sparse.linalg import norm
import random
import evaluation
from SVD_module import SVD


train2=recsys_utils.read_train()
test=recsys_utils.read_test_table()
truth=test['rating'].as_matrix()
user_map=recsys_utils.read_user_map()
movie_map=recsys_utils.read_movie_map()

print "Train Data Shape"
print train2.get_shape()

means_mat=train2.sum(axis=1)
counts=Counter(train2.nonzero()[0])


for i in range(means_mat.shape[0]):
		if i in counts.keys():
				means_mat[i,0]=means_mat[i, 0]/counts[i]
		else:
				means_mat[i,0]=0
                
train=deepcopy(train2)
nonzero_vals=np.array(np.nonzero(train))
nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])

for val in nonzero_vals:
    train[val[0], val[1]] -= means_mat[val[0]]
    

'''Calcuating Frobnieus Norm rowwise and column wise'''
forbenius_norm_matrix =  norm(train)
forbenius_norm_matrix_col =  norm(train,axis = 0)
forbenius_norm_matrix_row =  norm(train,axis = 1)

sum = 0
#print forbenius_norm_matrix_col[0]

''' Computing Probablities '''
for i in range(len(forbenius_norm_matrix_col)):
    forbenius_norm_matrix_col[i] = (forbenius_norm_matrix_col[i]/forbenius_norm_matrix)**2
    sum+=forbenius_norm_matrix_col[i]
 
print "Sum" , sum

''' Computing Probablities '''    
for i in range(len(forbenius_norm_matrix_row)):
    forbenius_norm_matrix_row[i] = (forbenius_norm_matrix_row[i]/forbenius_norm_matrix)**2
    sum+=forbenius_norm_matrix_row[i]
    
print "Sum" , sum
#print forbenius_norm_matrix_col[0] 

#print sum
#print forbenius_norm_matrix_col

#forbenius_norm_matrix_col.reshape(len(forbenius_norm_matrix_col),2)

"""This is No of rows to be selected which is equal to 4 * (no_of_dimension in svd) """
no_of_param = 200
print "No of Parameters Selected : ", no_of_param


def CUR_decompoaition_with_replacement(selected_Columns,selected_Rows):
    print "List of Selected Columns"
    print selected_Columns
    
    sel_frob_c = [forbenius_norm_matrix_col[i] for i in selected_Columns]
    
    sel_frob_r = [forbenius_norm_matrix_row[i] for i in selected_Rows]
    
    
    R_frob = np.column_stack((selected_Rows,sel_frob_r))
    C_frob = np.column_stack((selected_Columns,sel_frob_c))
    
    
    
    """Trying to convert the following list foramtion in certain function """
    
    #print range(len(forbenius_norm_matrix_row))
    print "No of Columns Considered : " ,len(C_frob)
     
    print "Matrix_C of CUR ::"
    #print C_frob[0]
    try_C = train[:,selected_Columns]
    print try_C.shape
    Matrix_C = [[((train[i,y[0]])/(sqrt(no_of_param*y[1])))for y in C_frob ]for i in range(len(forbenius_norm_matrix_row))]
    mat_c = np.array(Matrix_C)
    
    
    print len(Matrix_C)," , ",len(Matrix_C[0])
    
    R_frob = R_frob[:no_of_param]
    
    print "No of Rows Considered : " ,len(R_frob)
    
    print "Matrix_R of CUR"
    try_R = train[selected_Rows,:]
    print "Try_r"
    print try_R.shape
    Matrix_R = [[(train[int(y[0]),i])/(sqrt(no_of_param*y[1])) for i in range(len(forbenius_norm_matrix_col)) ]for y in R_frob]
    print len(Matrix_R)," , " , len(Matrix_R[0])
    #Matrix_C = [[((train[i[0],y[0]])/(sqrt(no_of_param*y[1]))) for i in R_frob]for y in C_frob]
    
    print"Matrix_W of CUR"
    W_matrix = [[train[int(i[0]),int(j[0])] for j in C_frob ]for i in R_frob]
    
    print "Calculating the SVD of W matrix"
    
    # X,Sig,Y = np.linalg.svd(W_matrix ,full_matrices = True ,  compute_uv = True )
    
    print "Shape of W matrix svd matrix"
    print X.shape , " ",Sig.shape , " ",Y.shape
    
    Psuedo_inv = Y.transpose()
    
    ''' Translating Sigma(1-d) to Sigma(Diagonal Matrix)  '''
    Sig_inv = np.diagflat(Sig)
    
    Sig_inv = np.linalg.inv(Sig_inv)
    
    Sig_inv = np.matmul(Sig_inv,Sig_inv)
    
    Psuedo_inv = np.matmul(Psuedo_inv , Sig_inv)
    Psuedo_inv = np.matmul(Psuedo_inv,X.transpose())
    
     
    mat_c = np.array(Matrix_C)
    mat_r = np.array(Matrix_R)
    print mat_c.shape, " ",Psuedo_inv.shape," ",mat_r.shape
    Cur_mat = np.matmul(Matrix_C,Psuedo_inv)
    Cur_mat = np.matmul(Cur_mat,Matrix_R)
    
    
    print "Final Matrix Shape with replacement as true"
    #print Cur_mat.shape
    
    print "Final Matrix Shape"
    print Cur_mat.shape
    
    print Cur_mat[0,0]
    
    Cur_mat = np.add(Cur_mat,means_mat)
    
    print "After adding mean"
    print Cur_mat[0,0]  
    
    evaluation.rmse(Cur_mat)    

    
""" Selecting random rows and columns based on their probablities"""
selected_Columns =  np.random.choice(len(forbenius_norm_matrix_col),no_of_param,replace = False, p = forbenius_norm_matrix_col)
selected_Rows = np.random.choice(len(forbenius_norm_matrix_row),no_of_param,replace = False , p = forbenius_norm_matrix_row)
selected_Columns.sort()
selected_Rows.sort()

selected_Columns1 =  np.random.choice(len(forbenius_norm_matrix_col),no_of_param,replace = True, p = forbenius_norm_matrix_col)
selected_Rows1 = np.random.choice(len(forbenius_norm_matrix_row),no_of_param,replace = True , p = forbenius_norm_matrix_row)
selected_Columns1.sort()
selected_Rows1.sort()

#CUR_decompoaition_with_replacement(selected_Columns1,selected_Rows1)

print "We are working with cloumns adn rows where repitition are not allowed"
print "List of Selected Columns"
print selected_Columns

sel_frob_c = [forbenius_norm_matrix_col[i] for i in selected_Columns]
sel_frob_r = [forbenius_norm_matrix_row[i] for i in selected_Rows]


R_frob = np.column_stack((selected_Rows,sel_frob_r))
C_frob = np.column_stack((selected_Columns,sel_frob_c))



"""Trying to convert the following list foramtion in certain function """

#print range(len(forbenius_norm_matrix_row))
print "No of Columns Considered : " ,len(C_frob)
 
print "Matrix_C of CUR ::"
#try_C = train[:,C_frob]
#print try_C.shape
print "Check"
Matrix_C = [[((train[i,y[0]])/(sqrt(no_of_param*y[1])))for y in C_frob ]for i in range(len(forbenius_norm_matrix_row))]
    


print len(Matrix_C)," , ",len(Matrix_C[0])

    
print "0,0 element of C matrix::" , Matrix_C[0][0]

R_frob = R_frob[:no_of_param]

print "No of Rows Considered : " ,len(R_frob)

print "Matrix_R of CUR"


Matrix_R = [[(train[int(y[0]),i])/(sqrt(no_of_param*y[1])) for i in range(len(forbenius_norm_matrix_col)) ]for y in R_frob]
print len(Matrix_R)," , " , len(Matrix_R[0])
#Matrix_C = [[((train[i[0],y[0]])/(sqrt(no_of_param*y[1]))) for i in R_frob]for y in C_frob]

print "0,0 element of R matrix::" , Matrix_R[0][0]

print"Matrix_W of CUR"
W_matrix = [[train[int(i[0]),int(j[0])] for j in C_frob ]for i in R_frob]


print "0,0 element of W matrix::" , W_matrix[0][0]


print "Calculating the SVD of W matrix"

# X,Sig,Y = np.linalg.svd(W_matrix ,full_matrices = True ,  compute_uv = True )
X, Sig, Y=SVD(W_matrix, 99.999)

print "Sigma Matrix :: " , Sig[0]

print "Shape of W matrix svd matrix"

print X.shape , " ",Sig.shape , " ",Y.shape

print "Matrix after svd XX"
print X[:10,:10]
print "Matrix after svd Sigma"
print Sig
print "Matrix after svd YY"

Sigma_sum = np.sum(Sig)
print Sigma_sum
Sigma_sum*=0.9999
x = 0
t = 0
for i in range(len(Sig)):
    t = i
    if x>Sigma_sum:
        break
    else:
        x+=Sig[i]
    
for i in range(t):
    X = np.delete(X,len(Sig)-1,1)
    Y = np.delete(Y,len(Sig)-1,0)
    Sig = np.delete(Sig,len(Sig)-1)
    
    
print "New Sigma Bro" , len(Sig)   
time.sleep(10) 
q = len(Sig)       
print Y[:10,:10]



Psuedo_inv = Y.transpose()

''' Translating Sigma(1-d) to Sigma(Diagonal Matrix)  '''
Sig_inv = np.diagflat(Sig)
print "E:: ",Sig_inv[q-1,q-1]
Sig_inv = np.linalg.inv(Sig_inv)
print "N:: ",Sig_inv[q-1,q-1]
time.sleep(10)

Sig_inv = np.matmul(Sig_inv,Sig_inv)

Psuedo_inv = np.matmul(Psuedo_inv , Sig_inv)
Psuedo_inv = np.matmul(Psuedo_inv,X.transpose())

 
mat_c = np.array(Matrix_C)
mat_r = np.array(Matrix_R)
print mat_c.shape, " ",Psuedo_inv.shape," ",mat_r.shape
Cur_mat = np.matmul(Matrix_C,Psuedo_inv)
Cur_mat = np.matmul(Cur_mat,Matrix_R)


print "Final Matrix Shape"
print Cur_mat.shape

a = Cur_mat[0,1]

Cur_mat = np.add(Cur_mat,means_mat)

print "After adding mean"
print (Cur_mat[0,1]-a),means_mat  

evaluation.rmse(Cur_mat)

evaluation.spearman_rank_correlation(Cur_mat)
























