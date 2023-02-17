import numpy as np # Just in case

# Import the classify and plotting functions
from classification_v2 import classify, cutout_plot

results = classify(full=True)

ID_pos=results[0]; RA_pos = results[1]; DEC_pos = results[2]
ID_neg=results[3];  RA_neg = results[4]; DEC_neg = results[5]

print("Writing positives/negatives to table...")

pos_idlist= '/data/des81.a/data/kherron/LSBG/Y6_FINAL/v3/y6_lsbg_FINAL'
np.save(pos_idlist,ID_pos)

neg_idlist= '/data/des81.a/data/kherron/LSBG/Y6_FINAL/v3/y6_negative_FINAL'
np.save(neg_idlist,ID_neg)

#Let's see how many were classified as LSBGs
print('Predicted to be LSBGs:',len(ID_pos))
print('Negative ones:',len(ID_neg))
     








