# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 20:59:26 2017

@author: Saurabh

"""

'''Importing Libraries'''
import numpy as np
import recsys_utils
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from time import time
from copy import deepcopy
from math import sqrt
from scipy.sparse.linalg import norm
import random
import evaluation 
# import SVD_module
from SVD import SVD

def Usr_Mean(train2):
    '''
    Calculate Mean of Every User 
    
    Inputs:
    train2 (2D numpy array): matrix from which the mean needs to be calculated.

    Returns:
    means_mat (1D numpy array): Matrix with row means 
    '''
    means_mat=train2.sum(axis=1)
    counts=Counter(train2.nonzero()[0])
    for i in range(means_mat.shape[0]):
        if i in counts.keys():
            means_mat[i,0]=means_mat[i, 0]/counts[i]
        else:
            means_mat[i,0]=0
    return means_mat

def Subtract_Mean_value(train2):
    '''
    Subtract row means from matrix mat_.
    Inputs:
    train2 (2D numpy array): matrix from which the mean needs to be subtracted.
    
    Returns:
    mat (1D numpy array): Array with row means.
    '''
    train=deepcopy(train2)
    nonzero_vals=np.array(np.nonzero(train))
    nonzero_vals= zip(nonzero_vals[0], nonzero_vals[1])

    for val in nonzero_vals:
        train[val[0], val[1]] -= means_mat[val[0]]
    return train
def calc_frob(train):
    '''
    Calculating Frobenius Sum of entire matrix and also row wise and column wise.
    Inputs:
    train (2D numpy array): matrix from which the Frobenius Sum needs to be calculated.
    
    Returns:
    forbenius_norm_matrix (double): Frobenius Sum of entire matrix
    forbenius_norm_matrix_col (1D numpy array): Array with Frobenius Sum of matrix Column Wise.
    forbenius_norm_matrix_row (1D numpy array): Array with Frobenius Sum of matrix Row Wise.
    '''
    forbenius_norm_matrix =  np.linalg.norm(train)
    forbenius_norm_matrix_col =  np.linalg.norm(train,axis = 0)
    forbenius_norm_matrix_row =  np.linalg.norm(train,axis = 1)  

    sum = 0
    
    
    ''' Computing Probablities '''
    for i in range(len(forbenius_norm_matrix_col)):
        forbenius_norm_matrix_col[i] = (forbenius_norm_matrix_col[i]/forbenius_norm_matrix)**2
        sum+=forbenius_norm_matrix_col[i]
     

    
    ''' Computing Probablities '''    
    for i in range(len(forbenius_norm_matrix_row)):
        forbenius_norm_matrix_row[i] = (forbenius_norm_matrix_row[i]/forbenius_norm_matrix)**2
        sum+=forbenius_norm_matrix_row[i]
        

    
    return forbenius_norm_matrix,forbenius_norm_matrix_col,forbenius_norm_matrix_row

def select(forbenius_norm_matrix_col,no_of_param,replace,forbenius_norm_matrix_row):
    '''
    Selcting Columns and rows based on Their Froebnius Norm and also randomly  
    
    Inputs:
    forbenius_norm_matrix_col (1D numpy array): matrix with Frobenius Sum Columns Wise.
    no_of_param (Integer) : No. of columns and rows to be selected 
    replace (bool) : Replace the column or row once it is selected 
    forbenius_norm_matrix_row (1D numpy array): matrix with Frobenius Sum Row Wise.
    
    Returns:
    selected_Columns (1D numpy array): List of selected Columns 
    selected_Rows (1 D Numpy array)  : List of selected rows
    
    '''

    if replace==False:
        selected_Columns =  np.random.choice(len(forbenius_norm_matrix_col),no_of_param,replace = False, p = forbenius_norm_matrix_col)
        selected_Rows = np.random.choice(len(forbenius_norm_matrix_row),no_of_param,replace = False , p = forbenius_norm_matrix_row)
        selected_Columns.sort()
        selected_Rows.sort()
    else:
        selected_Columns =  np.random.choice(len(forbenius_norm_matrix_col),no_of_param,replace = True, p = forbenius_norm_matrix_col)
        selected_Rows = np.random.choice(len(forbenius_norm_matrix_row),no_of_param,replace = True , p = forbenius_norm_matrix_row)
        selected_Columns.sort()
        selected_Rows.sort()

    return selected_Columns,selected_Rows


def Compute_U(train,C_frob,R_frob):
    '''
    Computing U matrix in CUR decomposition 

    Inputs:
    train (2D numpy array)  : matrix from which the W matrix need to be constructed.
    C_frob (2D numpy array) : List of selected Columns along with fronenius probability
    R_frob (2D numpy array) : List of selected Rows along with fronenius probability 
    
    Returns:
    U(2D Numpy Array)  : U Matrix Of CUR Decomposition 
    
    '''
    W_matrix = [[train[int(i[0]),int(j[0])] for j in C_frob ]for i in R_frob]




    X,Y,Sig1 = SVD(np.array(W_matrix))

    Sig = np.diag(Sig1)

    Sigma_sum = np.sum(Sig)
    print type(Sigma_sum)
    print Sigma_sum
    print Sigma_sum
    Sigma_sum*=0.9999
    x = 0
    t = 0
    for i in range(len(Sig)):
        t = i
        if x > Sigma_sum:
            break
        else:
            x+=Sig[i]

    print "Shape of X ::  Shape of Y" ,X.shape, " :: ", Y.shape

    for i in range(t):
        X = np.delete(X,len(Sig)-1,1)
        Y = np.delete(Y,len(Sig)-1,0)
        Sig = np.delete(Sig,len(Sig)-1)
        
    print "New Sigma Shape :: " , Sig.shape  
    print "New Sigma Bro" , len(Sig)   
    #time.sleep(10) 
    q = len(Sig)       

    print "Shape of X ::  Shape of Y" ,X.shape, " :: ", Y.shape



    Psuedo_inv = Y.transpose()

    ''' Translating Sigma(1-d) to Sigma(Diagonal Matrix)  '''

    print "New sigma :: " , Sig.shape
    Sig_inv = np.diagflat(Sig)


    print "E:: ",Sig_inv[q-1,q-1]
    Sig_inv = np.linalg.inv(Sig_inv)

    print "Sigma INverse :: " , Sig_inv.shape


    print "N:: ",Sig_inv[q-1,q-1]
    #time.sleep(10)

    Sig_inv = np.matmul(Sig_inv,Sig_inv.T)

    print "Sigma Shape :: " , Sig_inv.shape
    Psuedo_inv = np.matmul(Psuedo_inv , Sig_inv)
    Psuedo_inv = np.matmul(Psuedo_inv,X.transpose())
    U = Psuedo_inv
    return U

def Compute_Cur(Matrix_C,Matrix_R,U_mat):
    '''
    Reconstruct Original Matrix by Multiplying C*U*R

    Inputs:
    Matrix_C (2D numpy array)  : Matrix C of CUR Decomposition
    Matrix_R (2D numpy array) : Matrix R of CUR Decomposition
    U_mat (2D numpy array) : Matrix U of CUR Decomposition
    
    Returns:
    Cur_mat (2D numpy array) : Matrix Obtained by Multiplication of C*U*R components of CUR decomposition
    '''

    mat_c = np.array(Matrix_C)
    mat_r = np.array(Matrix_R)
    print mat_c.shape, " ",U_mat.shape," ",mat_r.shape
    Cur_mat = np.matmul(Matrix_C,U_mat)
    Cur_mat = np.matmul(Cur_mat,Matrix_R)


    print "Final Matrix Shape"
    print Cur_mat.shape

    a = Cur_mat[0,1]

    Cur_mat = np.add(Cur_mat,means_mat) 
    return Cur_mat

train2=recsys_utils.read_train()
test=recsys_utils.read_test_table()
truth=test['rating'].as_matrix()
user_map=recsys_utils.read_user_map()
movie_map=recsys_utils.read_movie_map()

print "Train Data Shape"
print train2.shape


means_mat = Usr_Mean(train2)


print "Done till here" 

train = Subtract_Mean_value(train2)         

    

'''Calcuating Frobnieus Norm rowwise and column wise'''
start_time_user=time()
forbenius_norm_matrix,forbenius_norm_matrix_col,forbenius_norm_matrix_row = calc_frob(train)


"""This is No of rows to be selected which is equal to 4 * (no_of_dimension in svd) """
no_of_param = 900
print "No of Parameters Selected : ", no_of_param


def CUR_decompoaition_with_replacement(selected_Columns,selected_Rows):
    print "List of Selected Columns"
    print selected_Columns
    
    sel_frob_c = [forbenius_norm_matrix_col[i] for i in selected_Columns]
    
    sel_frob_r = [forbenius_norm_matrix_row[i] for i in selected_Rows]
    
    
    R_frob = np.column_stack((selected_Rows,sel_frob_r))
    C_frob = np.column_stack((selected_Columns,sel_frob_c))
    
    
    
    """Trying to convert the following list foramtion in certain function """
    
    print "No of Columns Considered : " ,len(C_frob)
     
    print "Matrix_C of CUR ::"
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
    
    print"Matrix_W of CUR"
    W_matrix = [[train[int(i[0]),int(j[0])] for j in C_frob ]for i in R_frob]
    
    print "Calculating the SVD of W matrix"
    
    X,Y,Sig1= SVD(W_matrix)
    
    Sig = np.diag(Sig1)
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
    
    print "Final Matrix Shape"
    print Cur_mat.shape
    
    print Cur_mat[0,0]
    
    Cur_mat = np.add(Cur_mat,means_mat)
    

    
    evaluation.rmse(Cur_mat)    


""" Selecting random rows and columns based on their probablities"""

selected_Columns,selected_Rows = select(forbenius_norm_matrix_col,no_of_param,False,forbenius_norm_matrix_row)
selected_Columns1,selected_Rows1 = select(forbenius_norm_matrix_col,no_of_param,True,forbenius_norm_matrix_row)

def formMat_C(C_frob,train,no_of_param,forbenius_norm_matrix_row):
    Matrix_C = [[((train[i,int(y[0])])/(sqrt(no_of_param*y[1])))for y in C_frob ]for i in range(len(forbenius_norm_matrix_row))]
    return Matrix_C

print "We are working with cloumns adn rows where repitition are not allowed"
print "List of Selected Columns"
print selected_Columns

sel_frob_c = [forbenius_norm_matrix_col[i] for i in selected_Columns]
sel_frob_r = [forbenius_norm_matrix_row[i] for i in selected_Rows]


R_frob = np.column_stack((selected_Rows,sel_frob_r))
C_frob = np.column_stack((selected_Columns,sel_frob_c))



"""Trying to convert the following list foramtion in certain function """

print "No of Columns Considered : " ,len(C_frob)

Matrix_C = formMat_C(C_frob,train,no_of_param,forbenius_norm_matrix_row)    


print len(Matrix_C)," , ",len(Matrix_C[0])




Matrix_R = [[(train[int(y[0]),i])/(sqrt(no_of_param*y[1])) for i in range(len(forbenius_norm_matrix_col)) ]for y in R_frob]


U_mat = Compute_U(train,C_frob,R_frob)
 
Cur_mat = Compute_Cur(Matrix_C,Matrix_R,U_mat)


end_Time = time()


print "Time taken to execute CUR : " , (end_Time-start_time_user)

pred_Ratings = []
for idx,row in test.iterrows():
    pred_Ratings.append(Cur_mat[user_map[row['userId']] ,movie_map[row['movieId']]])
    
predictions = np.array(pred_Ratings)
print len(predictions)  
print "RMSE ERROR " , evaluation.RMSE(predictions,truth)
print "Spearman Rank Correlation  ", evaluation.spearman_rank_correlation(predictions,truth)
print "Top K rank Precision :: "   , evaluation.top_k_precision(predictions,test,np.squeeze(np.array(means_mat)),user_map) 

#evaluation.spearman_rank_correlation(Cur_mat)





    



















