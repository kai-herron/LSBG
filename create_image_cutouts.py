import numpy as np 
import pandas as pd
import urllib
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import astropy.io.fits as pyfits
from os.path import exists

# ====================================
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

# Read the file that contains RAs and DECs 
hdu = pyfits.open('/data/des80.b/data/burcinmp/y6_lsbg/y6/v2/y6_gold_2_0_lsb_skim.fits')
df= hdu[1].data

# Get coords
coadd_id_all = df['COADD_OBJECT_ID']
ra_all = df['RA']
dec_all = df['DEC']

#Which Subset of objects 
select_id =np.load('//data/des81.a/data/kherron/LSBG/DefaultResults/Defaultonly/defaultOnlyObs.npy')
ra =  [ra_all[ind] for ind, id in enumerate(coadd_id_all) if id in select_id] 
dec = [dec_all[ind] for ind, id in enumerate(coadd_id_all) if id in select_id]
coadd_id = [id for id in coadd_id_all if id in select_id]

dir0='/data/des81.a/data/kherron/LSBG/DefaultResults/Defaultonly/Images/DEF_ONLY'

# Calculate the length - the number of the candidates - it is going to be useful
N_cand = 1000 #len(ra)
print("Number of objects to check is:")
print(N_cand)


# Initialize array
Array = np.zeros([N_cand,64,64,3])

import time
from IPython.display import clear_output, display
import urllib.request
from urllib.error import HTTPError

zoom = 15

# Let's also time it
tim_in = time.time()

for i in range(N_cand): 
    #j = i #
    # Give a name to the figure. Name them as "Image_cand_(i).jpb
    # Where i is the number of the candidate
    # This is easy to change to ra, dec or coadd ID or whatever...
    j = coadd_id[i]
    fig_name = dir0+"Image_"+str(j)+".jpg"
    
    #Create now the name of the URL
    # This need to have as inputs (that change) the RA, DEC of each objec and zoom
    RA_loc = ra[i] #The RA of the i-th object
    DEC_loc = dec[i] # The DEC of the i-th object
    
    url_name = "http://legacysurvey.org/viewer/cutout.jpg?ra={0}&dec={1}&zoom={2}&layer=ls-dr9".format(RA_loc,DEC_loc,zoom)
    #url_name = "http://legacysurvey.org//viewer/jpeg-cutout?ra={0}&dec={1}&zoom={2}&layer=des-dr1".format(RA_loc,DEC_loc,zoom)
    #url_name = "https://www.legacysurvey.org//viewer/jpeg-cutout?ra={0}&dec={1}&layer=hsc2&zoom={2}".format(RA_loc,DEC_loc,zoom)
    #urllib.urlretrieve(url_name, fig_name) #Retrieves and saves each image
    #urllib.request.urlretrieve(url_name, fig_name) #Retrieves and saves each image
    req=urllib.request.Request(url_name)
    try:
        urllib.request.urlretrieve(url_name, fig_name)
    except HTTPError as e:
        content = e.read()

    if exists(dir0+'Image_'+str(j)+'.jpg'):
        image = Image.open(dir0+'Image_'+str(j)+'.jpg')
        # resize image
        new_image = image.resize((64, 64))
#    new_image = image.resize((60, 60))
    # Convert the image to an RGB array
        im_array = np.asarray(new_image)
    
        Array[i] = im_array
    
        clear_output(wait=True)
        print('runs:',i)
    # Leaving this here as an alternative way to do it
    #f = open(fig_name,'wb') #Open file and give name to save figure
    #f.write(urllib.urlopen(url_name).read()) #Open and read image from url
    #f.close() # Close the file
    
tim_fin = time.time()
print("Time to produce the figures (in minutes):")
print((tim_fin-tim_in)/60.0)

np.save('y6pos', Array)


