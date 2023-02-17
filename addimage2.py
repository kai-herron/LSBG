import numpy as np 
import pandas as pd
import urllib
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import astropy.io.fits as pyfits
from os.path import exists
import time
import time
from IPython.display import clear_output, display
import urllib.request
from urllib.error import HTTPError
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


#===============INPUT RA AND DEC SKIM FILE HERE==================
hdu = pyfits.open('/data/des80.b/data/burcinmp/y6_lsbg/y6/v2/y6_gold_2_0_lsb_skim.fits')
#================================================================
df= hdu[1].data

# Get coords
coadd_id_all = df['COADD_OBJECT_ID']
ra_all = df['RA']
dec_all = df['DEC']

import glob as glob
sample_direc = glob.glob("/data/des81.a/data/kherron/LSBG/Y6_FINAL/v3/Images/*sample*")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
failed = [0]
error = False
counter = 0
for i in sample_direc:
    images = glob.glob(i+"/*.png")      
    for j in images:
        if time.gmtime(os.path.getmtime(j))[1] == 2:
            counter +=1
            print("Image already done:", counter)
            continue
        else:
            coadd = int(os.path.splitext(os.path.basename(j))[0])
            find_coadd = np.isin(coadd_id_all,coadd)
            ra = ra_all[find_coadd]
            dec = dec_all[find_coadd]
            zoom = 12
            im = Image.open(j)
            im.save(j,"PNG")

            img0 = mpimg.imread(j)
            url_name = "http://legacysurvey.org/viewer/cutout.jpg?ra={0}&dec={1}&zoom={2}&layer=ls-dr10".format(ra[0],
                                                                                                        dec[0],
                                                                                                        zoom)
            try:
                req=urllib.request.Request(url_name)
                urllib.request.urlretrieve(url_name,'/data/des81.a/data/kherron/temp.png')
            except HTTPError as e:
                content = e.read()
            except:
                print("Error was encountered. Logging for redo...")
                counter += 1
                print("Image #",counter,"logged for redo.")
                error = True
                failed = np.append(failed,j)
                continue
            from PIL import Image
            im = Image.open('/data/des81.a/data/kherron/temp.png')
            im.save('/data/des81.a/data/kherron/temp.png', "PNG")

            img1 = mpimg.imread('/data/des81.a/data/kherron/temp.png')
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(img0)

            plt.xticks([])
            plt.yticks([])
            ax = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(img1)
            imgplot.set_clim(0.0, 0.7)

            plt.xticks([])
            plt.yticks([])
            plt.savefig(j, format='png')
            plt.close(fig)
            counter += 1
            print("Image #: ", counter, " done.")


        
        
if error == True:
    print("Beginning retries...")
    while len(failed != 0):
        j = failed[-1]
        coadd = int(os.path.splitext(os.path.basename(j))[0])
        find_coadd = np.isin(coadd_id_all,coadd)
        ra = ra_all[find_coadd]
        dec = dec_all[find_coadd]
        zoom = 12
        im = Image.open(j)
        im.save(j, "PNG")
        img0 = mpimg.imread(j)
        url_name = "http://legacysurvey.org/viewer/cutout.jpg?ra={0}&dec={1}&zoom={2}&layer=ls-dr10".format(ra[0],
                                                                                                    dec[0],
                                                                                                    zoom)
        try:
            req=urllib.request.Request(url_name)
            urllib.request.urlretrieve(url_name,'/data/des81.a/data/kherron/temp.png')
        except HTTPError as e:
            content = e.read()
        except:
            error = True
            failed = np.append(failed,j) 
        from PIL import Image
        im = Image.open('/data/des81.a/data/kherron/temp.png')
        im.save('/data/des81.a/data/kherron/temp.png', "PNG")

        img1 = mpimg.imread('/data/des81.a/data/kherron/temp.png')
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(img0)

        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(img1)
        imgplot.set_clim(0.0, 0.7)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(j, format='png')
        plt.close(fig)
print("Images done. Bye! :)")    