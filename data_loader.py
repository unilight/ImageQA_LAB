import numpy as np
import h5py
from os.path import join
from scipy import sparse

def load_image_feat(datadir):
	image_feat = None
	with h5py.File( join( datadir, 'hidden_oxford_mscoco.h5'),'r') as hf:
		key = 'hidden7'
		iwShape = hf[key + '_shape'][:]
		iwData = hf[key + '_data'][:]
		iwInd = hf[key + '_indices'][:]
		iwPtr = hf[key + '_indptr'][:]
		image_feat = sparse.csr_matrix((iwData, iwInd, iwPtr), shape=iwShape)
	return image_feat

