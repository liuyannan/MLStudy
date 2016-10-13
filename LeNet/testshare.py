import theano
import theano.tensor as T
import numpy

x=T.lmatrices('x')
y=T.lmatrices('y')

y = 2.0*x

func = theano.function([x],theano.gradient.jacobian(y[0],x))

t = func(numpy.asarray([[1,2],[3,4]]))

print t.shape

