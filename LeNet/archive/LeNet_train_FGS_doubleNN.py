from __future__ import print_function
import os
import sys
import timeit
import six.moves.cPickle as pickle
import gzip
import numpy
# import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from random import randint
import random
import theano.sandbox.cuda


class LeNetConvPoolLayer(object):
    '''Pool Layer of a convolutional network '''

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receive a gradient from
        # number out feature maps * filter height * fitler width / poolsize. because stride = 1
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.Wv = theano.shared(
            numpy.zeros(filter_shape, dtype=theano.config.floatX),
            borrow=True
        )

        # bias
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.bv = theano.shared(
            numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )
        # Convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsaple each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. We first reshape it to a tensor of shape(1, n_filters, 1, 1). Each bias
        # will thus be broadcasted across mini-batches and feature map width& height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layers
        self.params = [self.W, self.b]
        self.params_shape = [filter_shape, (filter_shape[0],)]
        self.velocity = [self.Wv, self.bv]

        # ?? keep track of model input
        self.input = input

class LeNetConvPoolLayer_OP(object):
    '''Pool Layer of a convolutional network '''

    def __init__(self, rng, input, filter_shape, image_shape, W, Wv, b, bv, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receive a gradient from
        # number out feature maps * filter height * fitler width / poolsize. because stride = 1
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

        # Convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsaple each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. We first reshape it to a tensor of shape(1, n_filters, 1, 1). Each bias
        # will thus be broadcasted across mini-batches and feature map width& height
        self.output = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layers
        self.params = [W, b]
        self.params_shape = [filter_shape, (filter_shape[0],)]
        self.velocity = [Wv, bv]

        # ?? keep track of model input        self.input = input

# namely the output layer
class FCSoftMaxLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, rng=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_bound = numpy.sqrt(6. / (n_in + 1))
        if rng is not None:
            self.W = theano.shared(
                value=numpy.asarray(
                    rng.uniform(
                        low=-W_bound,
                        high=W_bound,
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        self.Wv = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='Wv',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.bv = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bv',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.s_y_given_x = T.dot(input, self.W) + self.b

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        self.params_shape = [(n_in, n_out), (n_out,)]
        self.velocity = [self.Wv, self.bv]
        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def confidence_mean(self, y):
        return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class FCSoftMaxLayer_OP(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W, Wv, b, bv, rng=None):
        self.p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)
        self.s_y_given_x = T.dot(input, W) + b

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [W, b]
        self.params_shape = [(n_in, n_out), (n_out,)]
        self.velocity = [Wv, bv]
        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def confidence_mean(self, y):
        return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y is of the correct datatype
        return T.mean(T.neq(self.y_pred[0:(y.shape[0])], y))

class FCLayer_OP(object):
    def __init__(self, rng, input, n_in, n_out, W, b, Wv, bv,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        lin_output = T.dot(input, W) + b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [W, b]
        self.params_shape = [(n_in, n_out), (n_out,)]
        self.velocity = [Wv, bv]



class FCLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            Wv_values = numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
                Wv_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            Wv = theano.shared(value=Wv_values, name='Wv', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            bv_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            bv = theano.shared(value=bv_values, name='bv', borrow=True)

        self.W = W
        self.Wv = Wv
        self.b = b
        self.bv = bv

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.params_shape = [(n_in, n_out), (n_out,)]
        self.velocity = [self.Wv, self.bv]


def load_partdata(clist, dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            otrain_set, ovalid_set, otest_set = pickle.load(f, encoding='latin1')
        except:
            otrain_set, ovalid_set, otest_set = pickle.load(f)
    train_set = [[], []]
    valid_set = [[], []]
    test_set = [[], []]
    for i in range(0, len(otrain_set[1])):
        if otrain_set[1][i] in clist:
            train_set[0].append(otrain_set[0][i])
            train_set[1].append(otrain_set[1][i])
    for i in range(0, len(ovalid_set[1])):
        if ovalid_set[1][i] in clist:
            valid_set[0].append(ovalid_set[0][i])
            valid_set[1].append(ovalid_set[1][i])
    for i in range(0, len(otest_set[1])):
        if otest_set[1][i] in clist:
            test_set[0].append(otest_set[0][i])
            test_set[1].append(otest_set[1][i])

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class LeNet(object):
    def __init__(self, mu=0.5, learning_rate=0.1, n_epochs=40, dataset='mnist.pkl.gz', nkerns=[20, 50],
                 batch_size=250, gsf_epsilon=0.25, lam_contractive=10, lam_l2=0.001, train_divisor = 1):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        self.mu = mu
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.net_batch_size = batch_size
        self.train_divisor = train_divisor
        self.datasets = load_data(dataset)
        self.train_set_x, self.train_set_y = self.datasets[0]
        self.valid_set_x, self.valid_set_y = self.datasets[1]
        self.test_set_x, self.test_set_y = self.datasets[2]

        # compute number of minibatchs for train, valid, test
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches //= self.net_batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches //= self.net_batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches //= self.net_batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a minibatch

        # start-snippet-1
        x = T.matrix('x')
        y = T.ivector('y')

        # BUILD ACTUAL MODEL
        print('... building the model')

        # NN for GradX
        layer0_input = x.reshape((self.net_batch_size, 1, 28, 28))



        self.rng = numpy.random.RandomState(23455)
        self.layer0 = LeNetConvPoolLayer(self.rng, input=layer0_input, image_shape=(self.net_batch_size, 1, 28, 28),
                                         filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))


        layer1_input = self.layer0.output

        self.layer1 = LeNetConvPoolLayer(self.rng, input=layer1_input,
                                         image_shape=(self.net_batch_size, nkerns[0], 12, 12), \
                                         filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))


        # The FC layer. It operates on 2D matrices of shapece (batch_size, volumndepty*num_pixels). This will
        # generate a matrix of shape (batch_size, nkerns[1] * 4 * 4).
        # ????Hidden layer units happen to equal to minibatch?????
        layer2_input = self.layer1.output.flatten(2)

        self.layer2 = FCLayer(self.rng, input=layer2_input, n_in=nkerns[1] * 4 * 4, n_out=500, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer3_input = self.layer2.output
        self.layer3 = FCSoftMaxLayer(input=layer3_input, n_in=500, n_out=10, rng=self.rng)

        y_score_given_x = self.layer3.p_y_given_x
        # FGS Loss
        grad_predytox, _ = theano.scan(
            lambda ind, yi, y_score_given_x, x: (theano.gradient.jacobian(y_score_given_x[ind, yi], x))[ind, :]
            , sequences=[T.arange(self.net_batch_size), y],
            non_sequences=[y_score_given_x, x])

        guard_x = x - gsf_epsilon * T.sgn(grad_predytox)

        x_real = T.concatenate([x,guard_x],axis=0)
        y_real = T.concatenate([y,y],axis=0)


        # Real NN
        layer0_input_real = x_real.reshape((self.net_batch_size*2, 1, 28, 28))
        self.layer0_real = LeNetConvPoolLayer_OP(self.rng, input=layer0_input_real,
                                                 image_shape=(self.net_batch_size*2, 1, 28, 28),
                                                 filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2), W=self.layer0.W,
                                                 Wv=self.layer0.Wv, b=self.layer0.b, bv=self.layer0.bv)
        layer1_input_real = self.layer0_real.output
        self.layer1_real = LeNetConvPoolLayer_OP(self.rng, input=layer1_input_real,
                                                 image_shape=(self.net_batch_size*2, nkerns[0], 12, 12), \
                                                 filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2),
                                                 W=self.layer1.W, Wv=self.layer1.Wv, b=self.layer1.b, bv=self.layer1.bv)
        layer2_input_real = self.layer1_real.output.flatten(2)
        self.layer2_real = FCLayer_OP(self.rng, input=layer2_input_real, n_in=nkerns[1] * 4 * 4, n_out=500, activation=T.tanh, W=self.layer2.W, Wv=self.layer2.Wv, b = self.layer2.b, bv=self.layer2.bv)
        layer3_input_real = self.layer2_real.output
        self.layer3_real = FCSoftMaxLayer_OP(input=layer3_input_real, n_in=500, n_out=10, rng=self.rng, W=self.layer3.W, Wv=self.layer3.Wv, b = self.layer3.b, bv=self.layer3.bv)


        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        L2params = [self.layer3.W, self.layer2.W, self.layer1.W, self.layer0.W]
        self.params_shape = self.layer3.params_shape + self.layer2.params_shape + self.layer1.params_shape + self.layer0.params_shape
        velocities = self.layer3.velocity + self.layer2.velocity + self.layer1.velocity + self.layer0.velocity

        ############
        # Cost Function Definition
        ############
        cost = self.layer3_real.negative_log_likelihood(y_real)

        testnorm = T.sum((theano.gradient.jacobian(y_score_given_x[0], layer3_input)[:, 0, :]) ** 2) ** 0.5
        testgrads = T.grad(T.sum(testnorm), self.params)

        ########
        # Update Function
        ########

        grads = T.grad(cost, self.params)

        # momentum update
        updates = [(param_i, param_i - learning_rate * grad_i + mu * v_i)
                   for param_i, grad_i, v_i in zip(self.params, grads, velocities)]

        updates += [(v_i, mu * v_i - learning_rate * grad_i)
                    for grad_i, v_i in zip(grads, velocities)]

        # create a function to compute the mistakes that are made by the model
        self.validate_p = theano.function(
            [self.index],
            [testnorm] + testgrads,
            givens={
                x: self.valid_set_x[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        self.test_model = theano.function(
            [self.index],
            self.layer3_real.errors(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        self.validate_model = theano.function(
            [self.index],
            self.layer3_real.errors(y),
            givens={
                x: self.valid_set_x[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                y: self.valid_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        self.train_model = theano.function(
            [self.index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.train_set_y[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size]
            }
        )

        self.test_confidencefunc = theano.function(
            [self.index],
            self.layer3_real.confidence_mean(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

    def train(self):
        ##### TRAIN MODEL
        print('...training')
        patience = 10000
        patience_increase = 2

        improvement_threshold = 0.995
        validation_frequency = min(self.n_train_batches, patience // 2)

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(self.n_train_batches):
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if 0 <= iter <= 10:
                    # validate_value = self.validate_p(2)
                    # print(validate_value)
                    # compute zero-one loss on validation
                    validation_losses = [self.validate_model(i) for i in range(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.))
                if iter % 2 == 0:
                    print('training @ iter = ', iter)
                self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation
                    validation_losses = [self.validate_model(i) for i in range(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.))

                    # if we got the best validation score untile now
                    if this_validation_loss < best_validation_loss:

                        # improve patience
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it
                        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
                        test_conf = [self.test_confidencefunc(i) for i in range(self.n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%, test confidence is %f') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100., numpy.mean(test_conf)))
                if patience <= iter:
                    done_looping = True
                    break
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        return self.params


if __name__ == '__main__':
    # Build Mismatch Model
    theano.sandbox.cuda.use("gpu1")
    nn = LeNet(gsf_epsilon=0.25, n_epochs=40)
    params = nn.train()
    f = open('./weight/normal_weights_RandInit_FGS_doubleNN.pkl', 'wb')
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
