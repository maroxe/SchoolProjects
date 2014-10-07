import pickle
import numpy

import os


data_file = 'xFGBL20130702.pkl'

with open(data_file, 'rb') as input:
	r=pickle.load(input)
print r[:100]
		#do something with r


