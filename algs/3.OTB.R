# OTB: online to batch
# Goal: at every 1000 rounds, output a predictor and eveluate its loss 
# on a heldout dataset.

#----------
# IWAL-OTB: use h_It, the best after that round
# Need: h_It
# Compute on heldout data after algorithm finished.

#----------
# UCBHP-OTB1: weighted by pulled times, over non-requesters 
# UCBHP-OTB2: weighted by 1-mu, over non-requester
# UCBHP-OTB3: min mu among non-requesters

# Compute on heldout data along the algorithm.

# UCBHP-OTB Need: 
# 1. mu for all (h , r)
# 2. number of times pulled for all (h, r)
# 3. all_h, all_thre
