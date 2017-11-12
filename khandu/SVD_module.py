import numpy as np
from copy import deepcopy

def energy_calc(vec, percent_energy_retain):
	if vec.ndim==2:
		vec=np.squeeze(vec)
	vec=vec.astype(int)
	total_energy=np.sum(vec)
	required_energy=percent_energy_retain*total_energy/(100.0)
	index=np.argmin(vec.cumsum() < required_energy)+1
	print 'Total energy', total_energy
	print 'Required energy', required_energy
	print 'Index:', index
	ret_energy=np.sum(vec[0:index]) #temp, remove
	print 'Retained energy', ret_energy, ret_energy/float(total_energy)
	print 'Len:', len(vec), 'index:', index
	print 'Len:', len(vec), 'index:', index
	return index

def SVD(mat, percent_energy_retain=90, save_factorized=False):
	# Calculate U
	vals, vecs=np.linalg.eig(np.dot(mat, mat.T))
	vals=np.absolute(np.real(vals))

	no_eigenvalues=energy_calc(np.sort(vals)[::-1], percent_energy_retain)
	
	rank=no_eigenvalues #temp
	print 'Index=', rank
	indices=np.argsort(vals)[::-1][0:rank]
	U=np.real(vecs[:, indices])

	diag_vals=deepcopy(np.reshape(np.sqrt(np.sort(vals)[::-1])[0:rank], [rank]))

	# Calculate sigma
	sigma=np.zeros([rank, rank])
	np.fill_diagonal(sigma, diag_vals)

	#Calculate V
	V=np.zeros([mat.shape[1], rank])
	for i in range(rank):
		scaling_factor=(1/diag_vals[i])
		V[:, i]= scaling_factor*np.reshape(np.dot(mat.T, np.reshape(U[:, i], [U.shape[0], 1])), [mat.shape[1]])
	V_t=V.T

	if save_factorized:
		np.save('temp_data/U', U)
		np.save('temp_data/V_t', V_t)
		np.save('temp_data/sigma', sigma)
		print 'Matrices saved!'

	return U, V_t, sigma