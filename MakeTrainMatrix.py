#!/usr/bin/env python
# coding: utf-8

# Import basic packages
import numpy as np
import scipy as sp
import pandas as pd
import fitsio as ft

# ==========================================
# Matplotlib, urlib etc 
import urllib
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image





# ==========Read in the file================
FILE = ft.read('/data/des80.b/data/burcinmp/y6_lsbg/y6/v2/y6_gold_2_0_lsb_skim.fits')


#=====Make Training Matricies===============

mat = FILE

coadd_ids_in = mat['COADD_OBJECT_ID']
ras_in = mat['RA']
decs_in = mat['DEC']
A_IMAGE_in = mat['A_IMAGE']
B_IMAGE_in = mat['B_IMAGE']

MAG_AUTO_G_in = mat['MAG_AUTO_SFD_G']
FLUX_RADIUS_G_in = 0.263*mat['FLUX_RADIUS_G']
MU_EFF_MODEL_G_in = mat['MU_EFF_MODEL_SFD_G']
MU_MAX_MODEL_G_in = mat['MU_MAX_MODEL_SFD_G']
MU_MEAN_MODEL_G_in = mat['MU_MEAN_MODEL_SFD_G']

MAG_AUTO_R_in = mat['MAG_AUTO_SFD_R']
FLUX_RADIUS_R_in = 0.263*mat['FLUX_RADIUS_R']
MU_EFF_MODEL_R_in = mat['MU_EFF_MODEL_SFD_R']
MU_MAX_MODEL_R_in = mat['MU_MAX_MODEL_SFD_R']
MU_MEAN_MODEL_R_in = mat['MU_MEAN_MODEL_SFD_R']

MAG_AUTO_I_in = mat['MAG_AUTO_SFD_I']
FLUX_RADIUS_I_in = 0.263*mat['FLUX_RADIUS_I']
MU_EFF_MODEL_I_in = mat['MU_EFF_MODEL_SFD_I']
MU_MAX_MODEL_I_in = mat['MU_MAX_MODEL_SFD_I']
MU_MEAN_MODEL_I_in = mat['MU_MEAN_MODEL_SFD_I']

# Ellipticity
Ell_in = 1. - B_IMAGE_in/A_IMAGE_in

# Colors
col_g_r_in = MAG_AUTO_G_in - MAG_AUTO_R_in
col_g_i_in = MAG_AUTO_G_in - MAG_AUTO_I_in
col_r_i_in = MAG_AUTO_R_in - MAG_AUTO_I_in

# Length of matrix
len_n = len(ras_in)

# Initialize
X_mat_in = np.zeros([len_n,19])

# Populate
# Ellipticity 
X_mat_in[:,0] = Ell_in
# Colors
X_mat_in[:,1] = col_g_i_in
X_mat_in[:,2] = col_g_r_in;
X_mat_in[:,3] = col_r_i_in
# Magnitudes
X_mat_in[:,4] = MAG_AUTO_G_in
X_mat_in[:,5] = MAG_AUTO_R_in
X_mat_in[:,6] = MAG_AUTO_I_in
# Flux radii
X_mat_in[:,7] = FLUX_RADIUS_G_in
X_mat_in[:,8] = FLUX_RADIUS_R_in
X_mat_in[:,9] = FLUX_RADIUS_I_in
# Peak (max) surface brightness
X_mat_in[:,10] = MU_MAX_MODEL_G_in
X_mat_in[:,11] = MU_MAX_MODEL_R_in
X_mat_in[:,12] = MU_MAX_MODEL_I_in
# Effective surface brightness
X_mat_in[:,13] = MU_EFF_MODEL_G_in
X_mat_in[:,14] = MU_EFF_MODEL_R_in
X_mat_in[:,15] = MU_EFF_MODEL_I_in
# Mean surface brightness 
X_mat_in[:,16] = MU_MEAN_MODEL_G_in
X_mat_in[:,17] = MU_MEAN_MODEL_R_in
X_mat_in[:,18] = MU_MEAN_MODEL_I_in




np.save("/data/des81.a/data/kherron/LSBG/trainingfiles/training_ext/X_feat_eval.npy",
        X_mat_in)


    
    





