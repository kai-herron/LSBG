"""
Classifies objects (LSBGs vs Artifacts) given a skim-like fits file
"""

# Load packages
import numpy as np
import scipy 
from astropy.io import fits
import joblib
from joblib import dump, load

#Added from Alex....                                                                                                                                                                                                                                         
import sklearn
from sklearn import preprocessing
import pickle
import joblib
import fitsio
###Up to here...         

# **************************************************************
# ============ *** LOAD THE FITS FILE HERE *** =================
# Writing this in such a way that it loads DES Y6 skim =========
# Change the file name for simulations or so ===================

FILE = fits.open('/data/des80.b/data/burcinmp/y6_lsbg/y6/v2/y6_gold_2_0_lsb_skim.fits')

# ===== Load the scalers and classifiers =========
# ============================================================
# Load scalers
scaler_1 = joblib.load('/data/des81.a/data/kherron/LSBG/Default_Robust/scaler_v5.pkl') #1st stage  

# Load classifiers
cls_1 = joblib.load('/data/des81.a/data/kherron/LSBG/Default_Robust/classifier_v5.pkl') #1st stage

# ==============================================================
# ================= CREATE FEATURE MATRIX ======================
# Get the data 
data = FILE[1].data[:]

# Import COADD_ID, RA and DEC
coadd_id = data['COADD_OBJECT_ID']
ra = data['RA']
dec = data['DEC']

# ======= Write a function that creates a feature matrix =======
def feat_mat_around(ra_c=0.,dec_c=0.,degs=1.,full=True):
    """
    Function that returns arrays of coordinates 
    and a feature matrix to be used for classification,
    around given coordinates.
    -------------------------
    Input:
    ra_c - central coordinate/RA, default=0
    dec_c - central coordinate/DEC, default=0
    degs - how many degrees around the center to keep/default=1
    full - if True returns the feature matrix for the whole dataset
            otherwise only around a region
    -------------------------
    Output:
    ras_in - coordinates of objects that pass the criteria/RA
    decs_in - coordinates of objects that pass the criteria/DECs
    X_feat - feature matrix (unnormalized) of the objects that pass the criteria
    """
        
    # Define box of coordinates
    box = (ra>(ra_c-degs))&(ra<(ra_c+degs))&(dec>(dec_c-degs))&(dec<(dec_c+degs))
    
    
    # If full = False, then
    # Keep only coordinates & features within the above coordinate box
    if (full==False):
        data_in = data[box]
    else:
        data_in = data
        
    
    # Get different properties
    coadd_ids_in = data_in['COADD_OBJECT_ID']
    ras_in = data_in['RA']
    decs_in = data_in['DEC']
    A_IMAGE_in = data_in['A_IMAGE']
    B_IMAGE_in = data_in['B_IMAGE']
    MAG_AUTO_G_in = data_in['MAG_AUTO_G']
    FLUX_RADIUS_G_in = 0.263*data_in['FLUX_RADIUS_G']
    MU_EFF_MODEL_G_in = data_in['MU_EFF_MODEL_G']
    MU_MAX_G_in = data_in['MU_MAX_G']
    MU_MAX_MODEL_G_in = data_in['MU_MAX_MODEL_G']
    MU_MEAN_MODEL_G_in = data_in['MU_MEAN_MODEL_G']
    MAG_AUTO_R_in = data_in['MAG_AUTO_R']
    FLUX_RADIUS_R_in = 0.263*data_in['FLUX_RADIUS_R']
    MU_EFF_MODEL_R_in = data_in['MU_EFF_MODEL_R']
    MU_MAX_R_in = data_in['MU_MAX_R']
    MU_MAX_MODEL_R_in = data_in['MU_MAX_MODEL_R']
    MU_MEAN_MODEL_R_in = data_in['MU_MEAN_MODEL_R']
    MAG_AUTO_I_in = data_in['MAG_AUTO_I']
    FLUX_RADIUS_I_in = 0.263*data_in['FLUX_RADIUS_I']
    MU_EFF_MODEL_I_in = data_in['MU_EFF_MODEL_I']
    MU_MAX_I_in = data_in['MU_MAX_I']
    MU_MAX_MODEL_I_in = data_in['MU_MAX_MODEL_I']
    MU_MEAN_MODEL_I_in = data_in['MU_MEAN_MODEL_I']
    
    # =========== Create extra features ============
    # Ellipticity
    Ell_in = 1. - B_IMAGE_in/A_IMAGE_in
    
    # Colors
    col_g_r_in = MAG_AUTO_G_in - MAG_AUTO_R_in
    col_g_i_in = MAG_AUTO_G_in - MAG_AUTO_I_in
    col_r_i_in = MAG_AUTO_R_in - MAG_AUTO_I_in

    # ==============================================
    # ==============================================
    # ========= Create the feature matrix ==========
    
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
    # ========================================
    # ========================================
    # Return coordinates and matrix 
    
    return coadd_ids_in, ras_in, decs_in, X_mat_in



# ==========================================================
# ==========================================================
# ========== CLASSIFY ======================================

def classify(ra_c=0.,dec_c=0.,degs=1.,full=True,perc=0.2):
    """
    
    """
    
    # Get coordinates and feature matrix
    coadd_ids, ras, decs, X_mat = feat_mat_around(ra_c,dec_c,degs,full)
    
    # ========== First-stage classifier =======================
    # Standardize 
    X_mat_st_1st = scaler_1.transform(X_mat)
    
    # Predictions of the first classifier
    y_pred_1 = cls_1.predict(X_mat_st_1st)
    
    # Positives
    X_mat_pos_1 = X_mat[y_pred_1==1.]
    coadd_ids_pos_1 = coadd_ids[y_pred_1==1.]
    ras_pos_1 = ras[y_pred_1==1.]
    decs_pos_1 = decs[y_pred_1==1.]
    
    # Negatives
    coadd_ids_neg_1 = coadd_ids[y_pred_1==0.]
    ras_neg_1 = ras[y_pred_1==0.]
    decs_neg_1 = decs[y_pred_1==0.]
        
    arr = [coadd_ids_pos_1, ras_pos_1, decs_pos_1, coadd_ids_neg_1, ras_neg_1, decs_neg_1]        
    
    return arr

# =================================================================
# ==================== Plotting function ==========================
# =================================================================

# Import some packages here                                                                                                                                                                                                                           
# Matplotlib, urlib etc                                                                                                                                                                                                                               
import urllib
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image

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

def cutout_plot(ra_arr,dec_arr,start=0,end=100,save=True):
    """                                                                                                                                                                                                                                               
    This function gets the coordinates (ra,dec) of                                                                                                                                                                                                    
    a number of objects and outputs a figure with 100 cutouts                                                                                                                                                                                         
                                                                                                                                                                                                                                                      
    Inputs:                                                                                                                                                                                                                                           
    ra_arr - 1D array of RAs                                                                                                                                                                                                                          
    dec_arr - 1D array of DECs                                                                                                                                                                                                                        
    numb - where to start from, to creating the cutouts                                                                                                                                                                                               
    if 0, gives you the first 100 objects; if 100 gives you the next 100 etc                                                                                                                                                                          
    end: how many to plot, default = 100                                                                                                                                                                                                              
    save - whether to save the image or not, defualt = True                                                                                                                                                                                           
    """
    numba = int(end - start)

    # Initialize array                                                                                                                                                                                                                                
    Array = np.zeros([end,64,64,3])

    zoom=15
    # Populate array                                                                                                                                                                                                                                  
    for i in range(numba):
        j = i + start
        # Give a name to the figure. Name them as "Image_cand_(i).jpb                                                                                                                                                                                 
        # Where i is the number of the candidate                                                                                                                                                                                                      
        # This is easy to change to ra, dec or coadd ID or whatever...                                                                                                                                                                                
        fig_name = "Image_cand.jpg"

        #Create now the name of the URL                                                                                                                                                                                                               
        # This need to have as inputs (that change) the RA, DEC of each objec and zoom                                                                                                                                                                
        RA_loc = ra_arr[j] #The RA of the i-th object                                                                                                                                                                                                 
        DEC_loc = dec_arr[j] # The DEC of the i-th object                                                                                                                                                                                             

        url_name = "http://legacysurvey.org//viewer/jpeg-cutout?ra={0}&dec={1}&zoom={2}&layer=des-dr1".format(RA_loc,DEC_loc,zoom)
        #url_name = "https://www.legacysurvey.org//viewer/jpeg-cutout?ra={0}&dec={1}&layer=hsc2&zoom={2}".format(RA_loc,DEC_loc,zoom)                                                                                                                 
        urllib.request.urlretrieve(url_name, fig_name) #Retrieves and saves each image                                                                                                                                                                

        image = Image.open('Image_cand.jpg')
        # resize image                                                                                                                                                                                                                                
        new_image = image.resize((64, 64))
        # Convert the image to an RGB array                                                                                                                                                                                                           
        im_array = np.asarray(new_image)

        Array[i] = im_array

        clear_output(wait=True)
        print('runs:',i)

    # ==========================================                                                                                                                                                                                                      
    # Plot the cutouts generated in an array                                                                                                                                                                                                          
    n_rows = int(numba/5)
    n_cols = 5

    plt.figure(figsize=(4*n_cols*0.7, 4*n_rows*0.7))

    for i in range(n_rows*n_cols):
        #if (i==3):                                                                                                                                                                                                                                   
        #    plt.title("Matched objects",fontsize=25)                                                                                                                                                                                                 
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(Array[i]/255.)
        plt.axis('off')

    if (save==True):
        plt.savefig("Examples.pdf")

