#!/usr/bin/env python
# coding: utf-8

# Import basic packages
import numpy as np
import scipy as sp
import pandas as pd
from astropy.io import fits
import joblib
from joblib import dump, load

# ==== Scikit-learn =======================
# Preprocessing
from sklearn.preprocessing import StandardScaler #Standar scaler for standardization
from sklearn.preprocessing import RobustScaler #Robust scaler for high dispersion
from sklearn.model_selection import train_test_split # For random split

# Model selection
from sklearn.model_selection import GridSearchCV

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# ==========================================
# Matplotlib, urlib etc 
import urllib
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image
from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib', 'inline')
'exec(%matplotlib inline)'

# =========================================
# =========================================
# Adjust rc parameters to make plots pretty
def plot_pretty(dpi=200, fontsize=9):
    
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)       # dpi resolution of saved image files
    # if you have LaTeX installed on your laptop, uncomment the line below for prettier labels
    #plt.rc('text', usetex=True)      # use LaTeX to process labels
    plt.rc('font', size=fontsize)    # fontsize
    plt.rc('xtick', direction='in')  # make axes ticks point inward
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=10) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=10) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1]) # fix dotted lines

    return
plot_pretty()


# ## Training

# #### Import training dataset
# 
# Here we import the training dataset to be used in the classifier.
# 
# Specifically, we import the feature matrix that contains 19 features that are 
# the following (these are at the same order as the columns of the feature matrix):
# 
# - `Ellipticity`
# - `col_g_i`, `col_g_r`, `col_r_i` (Colors)
# - `MAG_AUTO_G`, `MAG_AUTO_R`, `MAG_AUTO_I` (magnitudes)
# - `FLUX_RADIUS_G`, `FLUX_RADIUS_R`, `FLUX_RADIUS_I` (flux radii)
# - `MU_MAX_MODEL_G`, `MU_MAX_MODEL_R`, `MU_MAX_MODEL_I` (max surface brightness)
# - `MU_EFF_MODEL_G`, `MU_EFF_MODEL_R`, `MU_EFF_MODEL_I` (effective surface brightness)
# - `MU_MEAN_MODEL_G`, `MU_MEAN_MODEL_R`, `MU_MEAN_MODEL_I` (mean surface brightness)
# 

# Feature matrix
X_feat_1st = np.load("/data/des80.b/data/burcinmp/y6_lsbg/y6/test_classifier/random_forest/v3/X_mat_v4_a.npy")

#Uncomment below to load in feature matrix for confirmed negative images, random sample, and Burcin's training

#===============================================================================================================
#randoms = np.load("/data/des81.a/data/kherron/LSBG/trainingfiles/Randoms/X_randoms_feat.npy")
#randoms_l = np.load("/data/des81.a/data/kherron/LSBG/trainingfiles/Randoms/y_randoms_lab.npy")

#burcin = np.load("/data/des80.b/data/burcinmp/y6_lsbg/y6/test_classifier/random_forest/v3/X_mat_v4_a.npy")
#burcin_l = np.load("/data/des80.b/data/burcinmp/y6_lsbg/y6/test_classifier/random_forest/v3/y_lab_v4_a.npy")

#sel = (burcin_l == 1)
#burcin = burcin[sel]
#burcin_l = burcin_l[sel]

#X_feat_1st = np.concatenate((burcin,randoms))
#===============================================================================================================

# Labels - 1: LSBGs, 0: artifacts
y_label_1st = np.load("/data/des80.b/data/burcinmp/y6_lsbg/y6/test_classifier/random_forest/v3/y_lab_v4_a.npy")


#Uncomment below to load in labels for confirmed negative images, random sample, and Burcin's training

#===============================================================================================================
#y_label_1st = np.concatenate((burcin_l,randoms_l))
#===============================================================================================================

# Train a standard scaler on the full dataset                                                                                                                                                                                                                               
eval_sample = np.load('/data/des81.a/data/kherron/LSBG/Default_Robust/X_eval_feat.npy')
combo = np.concatenate((X_feat_1st, eval_sample))
scaler_1st = StandardScaler().fit(combo)

# Standardize                                                                                                                                                                                                                                                               
X_feat_1st_st = scaler_1st.transform(X_feat_1st)

#### HERE YOU NEED TO MODIFY THE CLASSIFIER BASED ON YOUR TESTS!!!!
# Train the classifier here ***
RFC = RandomForestClassifier(n_estimators=100)

# Fit on the full dataset
RFC.fit(X_feat_1st_st, y_label_1st)

# ### Save the trained scalers and classifiers
# Save the scaler                                                                                                                                                                                                                                                     
joblib.dump(scaler_1st, '/data/des81.a/data/kherron/LSBG/Default_Robust/scaler_v5.pkl')

# Save the classifer                                                                                                                                                                                                                                                      
joblib.dump(RFC, '/data/des81.a/data/kherron/LSBG/Default_Robust/classifier_v5.pkl')


