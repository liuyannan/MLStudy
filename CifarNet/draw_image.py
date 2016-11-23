import matplotlib.pyplot as plt
import numpy
import cPickle as pickle

f = open('./cifar-10/data_batch_1', 'rb')
dset = pickle.load(f)
f.close()
theimage = dset['data'][104,:]

theimage = numpy.reshape(theimage,(3,32,32))
theimage = numpy.moveaxis(theimage,0,-1)
k = (theimage == 0)*1
plt.imshow(theimage)
plt.show()