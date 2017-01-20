# LeNet with weight mask

from __future__ import print_function
import os
import sys
import timeit
import six.moves.cPickle as pickle
import gzip
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from random import randint
import random
import theano.sandbox.cuda
import shutil


class IdenConvLayer(object):
    '''A ReLu convolutional Layer '''

    def __init__(self, rng, input, filter_shape, image_shape, border_m='valid'):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

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
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receive a gradient from
        # number out feature maps * filter height * fitler width  because stride = 1
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))

        # initialize weights with random weights
        W_bound = numpy.sqrt(2. / (fan_in ))
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

        self.Wmask = theano.shared(
            numpy.asarray(
                numpy.ones(filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # bias
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.bv = theano.shared(
            numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )
        self.bmask = theano.shared(
            numpy.ones((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )

        self.WM = self.W * self.Wmask

        # Convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.WM,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode= border_m
        )

        # # downsaple each feature map individually, using maxpooling
        # pooled_out = downsample.max_pool_2d(
        #     input=conv_out,
        #     ds=poolsize,
        #     ignore_border=True
        # )

        # add the bias term. We first reshape it to a tensor of shape(1, n_filters, 1, 1). Each bias
        # will thus be broadcasted across mini-batches and feature map width& height
        self.bM = self.b * self.bmask
        self.output = conv_out+self.bM.dimshuffle('x', 0, 'x', 'x')

        # store parameters of this layers
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask, self.bmask]
        # ?? keep track of model input
        self.input = input


class ReLuConvLayer(object):
    '''A ReLu convolutional Layer '''

    def __init__(self, rng, input, filter_shape, image_shape, border_m='valid'):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

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
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receive a gradient from
        # number out feature maps * filter height * fitler width  because stride = 1
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))

        # initialize weights with random weights
        W_bound = numpy.sqrt(2. / (fan_in ))
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

        self.Wmask = theano.shared(
            numpy.asarray(
                numpy.ones(filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # bias
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.bv = theano.shared(
            numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )
        self.bmask = theano.shared(
            numpy.ones((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )

        self.WM = self.W * self.Wmask

        # Convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.WM,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode= border_m
        )

        # # downsaple each feature map individually, using maxpooling
        # pooled_out = downsample.max_pool_2d(
        #     input=conv_out,
        #     ds=poolsize,
        #     ignore_border=True
        # )

        # add the bias term. We first reshape it to a tensor of shape(1, n_filters, 1, 1). Each bias
        # will thus be broadcasted across mini-batches and feature map width& height
        self.bM = self.b * self.bmask
        self.output = T.nnet.relu(conv_out+self.bM.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layers
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask, self.bmask]
        # ?? keep track of model input
        self.input = input


class MaxPoolingLayer(object):
    '''A MaxPooling Layer '''

    def __init__(self, input, poolsize=(2, 2), stride=None, ignore_border=True):
        """
        Allocate a MaxPoolingLayer with no parameters.

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        self.input = input
        self.output = downsample.max_pool_2d(
            input=input,
            ds=poolsize,
            st=stride,
            ignore_border=ignore_border
        )


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
        W_bound = numpy.sqrt(2. / (n_in + 1))
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
        self.Wmask = theano.shared(
            value=numpy.ones(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='Wmask',
            borrow=True
        )
        self.WM = self.W * self.Wmask
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
        self.bmask = theano.shared(
            value=numpy.ones(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='bmask',
            borrow=True
        )
        self.bM = self.b * self.bmask
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.exp(T.nnet.logsoftmax((T.dot(input, self.WM) + self.bM)))
        self.p_y_given_x_log = T.nnet.logsoftmax((T.dot(input, self.WM) + self.bM))
        self.logistic_regression = T.dot(input, self.WM) + self.bM
        self.s_y_given_x = T.dot(input, self.WM) + self.bM
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask, self.bmask]
        # keep track of model input
        self.input = input

    def confidence_mean(self, y):
        return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y])

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
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


# namely the output layer
class SoftMaxLayer(object):
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
        self.p_y_given_x = T.exp(T.nnet.logsoftmax(input))
        self.p_y_given_x_log = T.nnet.logsoftmax(input)
        self.logistic_regression = input
        self.s_y_given_x = input
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        self.input = input

    def confidence_mean(self, y):
        return T.mean(self.p_y_given_x[T.arange(y.shape[0]), y])

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
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class FCLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
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

            W = theano.shared(value=W_values, name='W', borrow=True)
            Wv = theano.shared(value=Wv_values, name='Wv', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            bv_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            bv = theano.shared(value=bv_values, name='bv', borrow=True)

        self.Wmask = theano.shared(numpy.ones((n_in, n_out), dtype=theano.config.floatX), name='Wmask', borrow=True)
        self.bmask = theano.shared(numpy.ones((n_out,), dtype=theano.config.floatX), name='bmask', borrow=True)

        self.W = W
        self.Wv = Wv
        self.b = b
        self.bv = bv
        self.WM = self.W * self.Wmask
        self.bM = self.b * self.bmask

        lin_output = T.dot(input, self.WM) + self.bM
        self.output = T.nnet.relu(lin_output)
        # parameters of the model
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask, self.bmask]


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here CIFAR)
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load training set
    train_x = []
    train_y = []
    for i in range(1,5):
        f = open(dataset+'/data_batch_'+str(i),'rb')
        dset = pickle.load(f)
        f.close()
        train_x.append(dset['data'])
        train_y += dset['labels']
    train_x = numpy.vstack(train_x)
    train_set = [train_x,train_y]

    # Load Validation set
    f = open(dataset + '/data_batch_5', 'rb')
    dset = pickle.load(f)
    f.close()
    valid_set = [dset['data'],dset['labels']]

    # Load test set
    f = open(dataset + '/test_batch', 'rb')
    dset = pickle.load(f)
    f.close()
    test_set = [dset['data'],dset['labels']]


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


class CifarNet(object):
    def __init__(self, initial_mu=0.9, initial_learning_rate=0.01, n_epochs=80, dataset='./cifar-10',
                 nkerns=[64, 64, 128, 128, 256, 256],
                 batch_size=128, lam_l2=0.001, train_divisor=1, cf_type='L2', lam_contractive=100, phase='Train'):
        """ CifarNet on Cifar10 dataset

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
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.net_batch_size = batch_size
        self.train_divisor = train_divisor
        self.datasets = load_data(dataset)
        self.train_set_x, self.train_set_y = self.datasets[0]
        self.valid_set_x, self.valid_set_y = self.datasets[1]
        self.test_set_x, self.test_set_y = self.datasets[2]

        # compute number of minibatchs for train, valid, test
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches //= (self.net_batch_size / train_divisor)
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches //= self.net_batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches //= self.net_batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a minibatch

        # start-snippet-1
        x = T.matrix('x')
        y = T.ivector('y')
        tclass = T.lscalar('tclass')

        #Dropout Switch
        switch_train = T.iscalar('switch_train')

        # BUILD ACTUAL MODEL
        print('... building the model')
        self.rng = numpy.random.RandomState(234567)
        self.srng = RandomStreams(self.rng.randint(1239))
        # Reshape matrix of rasterized image of shape(batch_size, 32* 32)  to 4D tensor
        layer0_input = x.reshape((self.net_batch_size, 3, 32, 32))

        # First ReLuConv layer:
        # Filtering reduces the image size to(32-3+1, 32-3+1) = (30, 30)
        # 4D output tensor is thus of shape (batchsize, nkerns[0],30,30)
        self.layer0 = ReLuConvLayer(self.rng, input=layer0_input, image_shape=(self.net_batch_size, 3, 32, 32),
                                    filter_shape=(nkerns[0], 3, 3, 3))

        # Second ReLuConv layer:
        # output image size to (30-3+1, 30-3+1)=(28,28)
        # 4D output tensor is (batchsize, nkerns[1], 28, 28)
        layer1_input = self.layer0.output
        self.layer1 = ReLuConvLayer(self.rng, input=layer1_input, image_shape=(self.net_batch_size, nkerns[0], 30, 30),
                                    filter_shape=(nkerns[1], nkerns[0], 3, 3))

        # Maxpooling layer:
        # Output size is (batchsize, nkerns[1], 14, 14)
        self.layer2 = MaxPoolingLayer(self.layer1.output, poolsize=(2, 2))

        # Third ReLuConv layer:
        # output image size to (14-3+1, 14-3+1)=(12,12)
        # 4D output tensor is (batchsize, nkerns[2], 12, 12)
        layer3_input = self.layer2.output
        self.layer3 = ReLuConvLayer(self.rng, input=layer3_input, image_shape=(self.net_batch_size, nkerns[1], 14, 14),
                                    filter_shape=(nkerns[2], nkerns[1], 3, 3))

        # Fourth ReLuConv layer:
        # output image size to (12-3+1, 12-3+1)=(10,10)
        # 4D output tensor is (batchsize, nkerns[3], 10, 10)
        layer4_input = self.layer3.output
        self.layer4 = ReLuConvLayer(self.rng, input=layer4_input, image_shape=(self.net_batch_size, nkerns[2], 12, 12),
                                    filter_shape=(nkerns[3], nkerns[2], 3, 3))

        # Maxpooling layer:
        # Output size is (batchsize, nkerns[3], 5, 5)
        self.layer5 = MaxPoolingLayer(self.layer4.output, poolsize=(2, 2))

        # The First FC layer. It operates on 2D matrices of shapece (batch_size, volumndepty*num_pixels). This will
        # generate a matrix of shape (batch_size, nkerns[3] * 5 * 5).
        layer6_input = self.layer5.output.flatten(2)
        self.layer6 = FCLayer(self.rng, input=layer6_input, n_in=nkerns[3] * 5 * 5, n_out=nkerns[4])

        # dropout for Layer6
        layer6_drop = self.srng.binomial(size=self.layer6.output.shape,p=0.5,dtype=theano.config.floatX)* self.layer6.output
        layer6_output = T.switch(T.neq(switch_train,1),0.5*self.layer6.output,layer6_drop)


        # The Second Fc Layer
        layer7_input = layer6_output
        self.layer7 = FCLayer(self.rng, input=layer7_input, n_in=nkerns[4], n_out=nkerns[5])

        # dropout for layer7
        layer7_drop = self.srng.binomial(size=self.layer7.output.shape,p=0.5,dtype=theano.config.floatX)* self.layer7.output
        layer7_output = T.switch(T.neq(switch_train,1),0.5*self.layer7.output,layer7_drop)


        # classify the values of the fully-connected ReLu layer
        layer8_input = layer7_output
        self.layer8 = FCSoftMaxLayer(input=layer8_input, n_in=nkerns[5], n_out=10, rng=self.rng)

        self.params = self.layer8.params + self.layer7.params + self.layer6.params + self.layer4.params + self.layer3.params + self.layer1.params + self.layer0.params
        self.masks = self.layer8.mask + self.layer7.mask + self.layer6.mask + self.layer4.mask + self.layer3.mask + self.layer1.mask + self.layer0.mask
        L2params = [self.layer8.W, self.layer7.W, self.layer6.W, self.layer4.W, self.layer3.W, self.layer1.W,
                    self.layer0.W]
        self.velocities = self.layer8.velocity + self.layer7.velocity + self.layer6.velocity + self.layer4.velocity + self.layer3.velocity + self.layer1.velocity + self.layer0.velocity
        self.log_init()
        ############
        # Cost Function Definition
        ############

        #TODO: input shared variable

        paramssum = T.sum(T.sqr(L2params[0]))
        for i in range(1, len(L2params)):
            paramssum += T.sum(T.sqr(L2params[i]))

        regularization = lam_l2 * paramssum

        delta_L_to_x = T.grad(self.layer8.negative_log_likelihood(y), x)
        delta_norm = T.sum(delta_L_to_x ** 2) / T.shape(x)[0]

        if cf_type == 'L2':
            cost = self.layer8.negative_log_likelihood(y) + regularization
        elif cf_type == 'Contract_Likelihood':
            cost = self.layer8.negative_log_likelihood(y) + lam_contractive * (delta_norm)
        elif cf_type == 'no_regular':
            cost = self.layer8.negative_log_likelihood(y)

        ########
        # Update Function
        ########

        grads = T.grad(cost, self.params)

        learning_rate = theano.shared(numpy.cast[theano.config.floatX](initial_learning_rate), name='lr')
        mu = theano.shared(numpy.cast[theano.config.floatX](initial_mu), name='momentum')
        self.learning_rate = learning_rate
        self.mu = mu
        # momentum update
        updates = [(param_i, param_i - learning_rate * grad_i + mu * v_i)
                   for param_i, grad_i, v_i in zip(self.params, grads, self.velocities)]

        updates += [(v_i, mu * v_i - learning_rate * grad_i)
                    for grad_i, v_i in zip(grads, self.velocities)]


        self.test_contractive = theano.function(
            [self.index],
            delta_norm,
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            }
        )

        self.test_model = theano.function(
            [self.index],
            self.layer8.errors(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            }
        )

        self.test_grads_to_params = theano.function(
            [self.index],
            grads,
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            }
        )

        self.validate_model = theano.function(
            [self.index],
            self.layer8.errors(y),
            givens={
                x: self.valid_set_x[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                y: self.valid_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            }
        )


        self.train_model = theano.function(
            [self.index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.train_set_y[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](1)
            }
        )

        self.test_confidencefunc = theano.function(
            [self.index],
            self.layer8.confidence_mean(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            }
        )

        ########
        # Adversarial related functions
        ########
        score = self.layer8.p_y_given_x_log[0][tclass]
        self.proby = theano.function([x, tclass], [score, self.layer8.y_pred], allow_input_downcast=True, givens={switch_train:numpy.cast['int32'](0)})
        class_grads = T.grad(score, x)
        self.ygradsfunc = theano.function([x, tclass], class_grads, allow_input_downcast=True, givens={switch_train:numpy.cast['int32'](0)})

    def get_grad_and_proby_func(self):
        return self.ygradsfunc, self.proby

    def log_cur_params(self):
        self.cur_params = []
        self.cur_velocities = []
        for i in range(len(self.params)):
            self.cur_params.append(self.params[i].get_value())
            self.cur_velocities.append(self.velocities[i].get_value())

    def recover_params(self):
        for i in range(len(self.params)):
            self.params[i].set_value(self.cur_params[i])
            self.velocities[i].set_value(self.cur_velocities[i])

    def log_init(self):
        self.params_init = []
        for i in range(len(self.params)):
            self.params_init.append(self.params[i].get_value())

    def init_by_log(self):
        for i in range(len(self.params)):
            self.params[i].set_value(self.params_init[i])
            vel_shape = self.velocities[i].get_value().shape
            self.velocities[i].set_value(numpy.zeros(vel_shape,dtype=theano.config.floatX))

    def train(self, n_epochs):
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

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in range(self.n_train_batches):
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if 0 <= iter <= 10:
                    validation_losses = [self.validate_model(i) for i in range(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch, minibatch_index + 1, self.n_train_batches, this_validation_loss * 100.))
                # if iter % 2 == 0:
                #     print('training @ iter = ', iter)
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
            if epoch%10 == 0:
                new_lr = self.learning_rate.get_value()*0.95
                self.learning_rate.set_value(numpy.cast[theano.config.floatX](new_lr))
                new_mu = self.mu.get_value()*0.5
                self.mu.set_value(numpy.cast[theano.config.floatX](new_mu))
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        grads_to_pp_batch = self.test_grads_to_params(0)
        return self.params, self.masks, test_score, grads_to_pp_batch

    def connection_count(self):
        flatten_mask = 0
        for i in range(len(self.params)):
            normal_group = numpy.sum(self.masks[i].get_value())
            flatten_mask += normal_group
        return flatten_mask

    def resume_all(self, params, masks):
        for i in range(len(self.params)):
            self.params[i].set_value(params[i].get_value())
            self.masks[i].set_value(masks[i].get_value())

    def resume_mask(self, masks):
        for i in range(len(self.params)):
            self.masks[i].set_value(masks[i].get_value())

    def inject_fault(self, probability):
        for i in range(len(self.params)):
            mask_group = self.masks[i].get_value()
            group_shape = mask_group.shape
            mask_group = numpy.reshape(mask_group, (-1))
            for ind in range(len(mask_group)):
                if mask_group[ind] == 1:
                    mask_group[ind] = numpy.random.binomial(1, 1 - probability)
            mask_group.resize(group_shape)
            self.masks[i].set_value(mask_group)
        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
        test_score = numpy.mean(test_losses)
        return test_score

    def get_accuracy(self):
        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
        test_score = numpy.mean(test_losses)
        return test_score

def find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, image, target_class):
    transform_img = [numpy.copy(image)]
    conflist = []
    epoch = 0
    last_step_size = 0.
    while epoch < 1000:
        probyvalue, cur_y= proby(transform_img, target_class)
        probyvalue = numpy.exp(probyvalue)
        gradsvalue = gradsfunc(transform_img, target_class)
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5

        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e, grad is %e, and last step is %e" % (epoch, probyvalue, gradmag, last_step_size))
            conflist.append(probyvalue)

        # if probyvalue > 0.55:
        if cur_y == target_class:
            # if probyvalue > 0.99 and gradmag < 1e-4:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))
            break

        p = gradsvalue.reshape(32*32*3, 1)
        p *= 0.1 / gradmag
        t = p.T.dot(p) * 0.
        #if last_step_size >= 500.:
        step_size = 40.
        #else:
            #step_size = last_step_size + 50
        #if epoch >= 200:
            #step_size = 400.

        f_x_k = numpy.exp(proby(transform_img, target_class)[0])
        upper_bound_img = numpy.ones((32*32*3,))*255
        lower_bound_img = numpy.zeros((32*32*3,))
        predict_img = transform_img[0] + step_size * p.reshape(32*32*3, )
        predict_img = numpy.maximum(lower_bound_img, numpy.minimum(predict_img, upper_bound_img))
        btcount = 0
        while numpy.exp(proby([predict_img], target_class)[0]) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(32*32*3, )
            predict_img = numpy.maximum(lower_bound_img, numpy.minimum(predict_img, upper_bound_img))
            step_size *= 0.8
            btcount += 1
            if step_size < 1e-6:
                predict_img = transform_img[0] + p.reshape(32*32*3, )
                break
            #print("backtracking count is %d"%btcount)
        last_step_size = step_size

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img, cur_y]


def dump_mnist(fname, gradsfunc, proby,folder):
    # CIFAR
    dataf = open('./cifar-10/test_batch', 'rb')
    dataset = pickle.load(dataf)
    dataf.close()
    test_x = dataset['data']
    test_y = dataset['labels']
    confidence_data = []
    for i in range(100):
        print("evaluate CIFAR10 %i" % i)
        target_class_list = range(10)
        target_class_list.remove(test_y[i])
        for target_class in target_class_list:
            ori_img = numpy.asarray(test_x[i,:], dtype=theano.config.floatX)
            confidence_list, reimg, cur_y = find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, ori_img,
                                                                                  target_class)
            if len(confidence_list)==0 and len(reimg) == 0:
                continue
            confidence_data.append([[test_y[i], target_class, cur_y ], [ori_img, reimg], confidence_list])
    f = open('./eval_efforts_rough/Constraint_mnist_GDBack_Compression_'+folder+'_' + fname + '.pkl', 'wb')
    pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()



def remove_weights_global(nn,ratio,policy):

    if policy in['CC','CCV2']:
        L2xp_batch = nn.L2xp(0)
        for train_i in range(1,nn.n_train_batches):
            tmp_batch = nn.L2xp(train_i)
            L2xp_batch = [L2xp_batch[ind]+tmp_batch[ind] for ind in range(len(L2xp_batch))]
        L2xp_batch = [ numpy.asarray(x)/nn.n_train_batches for x in L2xp_batch]
        L2xp_sign = [numpy.sign(x) for x in L2xp_batch]
    elif policy == 'OBD':
        Sec_Gradients = nn.SecondOrderGradient(randint(0, nn.n_train_batches - 1))

    # Sort different parameters according to metric
    flatten_params = []
    for i in range(0, len(nn.params), 2):
        if policy == 'SW':
            params_group = nn.params[i].get_value()
        elif policy == 'CC':
            params_group = -1 * nn.params[i].get_value() * L2xp_sign[i]
        elif policy == 'CCV2':
            params_group = numpy.absolute(nn.params[i].get_value()) / numpy.absolute(L2xp_batch[i])
        elif policy == 'OBD':
            params_group = numpy.square(nn.params[i].get_value())*Sec_Gradients[i]
        else:
            print("Remove Edge Policy not FOUND!!!")

        # Remove already masked parameters
        mask_group = nn.masks[i].get_value()
        cur_flatten_params = list(numpy.reshape(params_group,(-1)))
        cur_flatten_mask = list(numpy.reshape(mask_group,(-1)))
        zip_pm = zip(cur_flatten_params,cur_flatten_mask)
        zip_pm = filter(lambda x: x[1] == 1, zip_pm)
        if len(zip_pm) == 0:
            continue
        flatten_params += list(zip(*zip_pm)[0])

    # Find the threshold
    if policy == 'SW':
        # flatten_params = filter(lambda a: a != 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()
    elif policy == 'CC':
        flatten_params = filter(lambda a: a < 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()
    elif policy in ['OBD','CCV2']:
        flatten_params.sort()

    effective_connection_count = len(flatten_params)
    remove_edge_number = int(effective_connection_count * ratio) - 1
    if remove_edge_number <= 0:
        return
    thval = flatten_params[remove_edge_number]


    # Remove Weights
    for i in range(0, len(nn.params), 2):
        if policy == "SW":
            params_group = nn.params[i].get_value()
            remove_edge_index = abs(params_group) < thval
        elif policy == 'CC':
            params_group = -1 * nn.params[i].get_value() * L2xp_sign[i]
            remove_edge_index = numpy.logical_and(params_group<0,params_group>(-1*thval))
        elif policy == 'OBD':
            params_group = numpy.square(nn.params[i].get_value()) * Sec_Gradients[i]
            remove_edge_index = params_group <= thval
        elif policy == 'CCV2':
            params_group = numpy.absolute(nn.params[i].get_value()) / numpy.absolute(L2xp_batch[i])
            remove_edge_index = params_group < thval

        mask_group = nn.masks[i].get_value()
        mask_group[remove_edge_index] = 0
        nn.masks[i].set_value(mask_group)



def remove_weights_by_layer(nn,ratio,policy):

    if policy in['CC','CCV2']:
        L2xp_batch = nn.L2xp(0)
        for train_i in range(1, nn.n_train_batches):
            tmp_batch = nn.L2xp(train_i)
            L2xp_batch = [L2xp_batch[ind] + tmp_batch[ind] for ind in range(len(L2xp_batch))]
        L2xp_batch = [x / nn.n_train_batches for x in L2xp_batch]
        L2xp_sign = [numpy.sign(x) for x in L2xp_batch]
    elif policy == 'OBD':
        Sec_Gradients = nn.SecondOrderGradient(randint(0, nn.n_train_batches - 1))

    # Sort different parameters according to metric
    for i in range(0, len(nn.params), 2):
        flatten_params = []
        if policy == 'SW':
            params_group = nn.params[i].get_value()
        elif policy == 'CC':
            params_group = -1 * nn.params[i].get_value() * L2xp_sign[i]
        elif policy == 'CCV2':
            params_group = numpy.absolute(nn.params[i].get_value()) / numpy.absolute(L2xp_batch[i])
        elif policy == 'OBD':
            params_group = numpy.square(nn.params[i].get_value())*Sec_Gradients[i]
        else:
            print("Remove Edge Policy not FOUND!!!")

        # Remove already masked parameters
        mask_group = nn.masks[i].get_value()
        cur_flatten_params = list(numpy.reshape(params_group,(-1)))
        cur_flatten_mask = list(numpy.reshape(mask_group,(-1)))
        zip_pm = zip(cur_flatten_params,cur_flatten_mask)
        zip_pm = filter(lambda x: x[1] == 1, zip_pm)
        flatten_params += list(zip(*zip_pm)[0])

        # Find the threshold
        if policy == 'SW':
            # flatten_params = filter(lambda a: a != 0, flatten_params)
            flatten_params = map(lambda x: abs(x), flatten_params)
            flatten_params.sort()
        elif policy == 'CC':
            flatten_params = filter(lambda a: a < 0, flatten_params)
            flatten_params = map(lambda x: abs(x), flatten_params)
            flatten_params.sort()
        elif policy in ['OBD','CCV2']:
            flatten_params.sort()

        effective_connection_count = len(flatten_params)
        remove_edge_number = int(effective_connection_count * ratio) - 1
        if remove_edge_number <= 0:
            return
        thval = flatten_params[remove_edge_number]


        # Remove Weights
        if policy == "SW":
            remove_edge_index = abs(params_group) < thval
        elif policy == 'CC':
            remove_edge_index = numpy.logical_and(params_group<0,params_group>(-1*thval))
        elif policy == 'OBD':
            remove_edge_index = params_group <= thval
        elif policy == 'CCV2':
            remove_edge_index = params_group < thval

        mask_group = nn.masks[i].get_value()
        mask_group[remove_edge_index] = 0
        nn.masks[i].set_value(mask_group)




def remove_weights_SWandCT(nn,ratio):

    # contract_term for each parameter
    L2xp_batch = nn.L2xp(0)
    for train_i in range(1,nn.n_train_batches):
        tmp_batch = nn.L2xp(train_i)
        L2xp_batch = [L2xp_batch[ind]+tmp_batch[ind] for ind in range(len(L2xp_batch))]
    L2xp_batch = [ numpy.asarray(x)/nn.n_train_batches for x in L2xp_batch]
    L2xp_sign = [numpy.sign(x) for x in L2xp_batch]


    # Stage one: Select small weights edge
    # Sort different parameters according to metric
    flatten_params = []
    for i in range(0,len(nn.params),2):
        params_group = nn.params[i].get_value()

        # Remove already masked parameters
        mask_group = nn.masks[i].get_value()
        cur_flatten_params = list(numpy.reshape(params_group,(-1)))
        cur_flatten_mask = list(numpy.reshape(mask_group,(-1)))
        zip_pm = zip(cur_flatten_params,cur_flatten_mask)
        zip_pm = filter(lambda x: x[1] == 1, zip_pm)
        if len(zip_pm) == 0:
            continue
        flatten_params += list(zip(*zip_pm)[0])

    # Find the threshold for small weight
    flatten_params = map(lambda x: abs(x), flatten_params)
    flatten_params.sort()
    effective_connection_count = len(flatten_params)
    candidate_edge_number = int(effective_connection_count * ratio * 2) - 1
    sw_thval = flatten_params[candidate_edge_number]

    # Stage two: find small weight edges that degrade contractive terms
    flatten_ct = []
    for i in range(0,len(nn.params),2):
        params_group = nn.params[i].get_value()
        sw_edge_index = abs(params_group) <= sw_thval
        ct_change = -1 * params_group * L2xp_batch[i]
        mask_group = nn.masks[i].get_value()
        existing_edge_index = mask_group == 1
        flatten_ct += list(ct_change[numpy.logical_and(sw_edge_index,existing_edge_index)])

    flatten_ct.sort()
    ct_thval = flatten_ct[int((len(flatten_ct)-1)*0.5)]


    for i in range(0,len(nn.params),2):
        params_group = nn.params[i].get_value()
        sw_edge_index = abs(params_group) <= sw_thval
        ct_change = -1 * params_group * L2xp_batch[i]
        ct_edge_index = ct_change <= ct_thval
        remove_edge_index = numpy.logical_and(sw_edge_index,ct_edge_index)
        mask_group = nn.masks[i].get_value()
        mask_group[remove_edge_index] = 0
        nn.masks[i].set_value(mask_group)



def remove_weights(nn,ratio,policy,LBL=False):
    if 'CCV3' in policy :
        remove_weights_SWandCT(nn,ratio)
        return

    if LBL:
        remove_weights_by_layer(nn,ratio,policy)
    else:
        remove_weights_global(nn,ratio,policy)


def compression_API(folder, cf, rm_policy='SW',ratio=0.1,resume=False, nn=None):
    '''
    Compress the network by eliminating small weight parameters and less contractive parameters in turn.
    :return:
    '''
    if nn is None:
        nn = CifarNet(cf_type=cf)

    # Continue compression from the existing smallest structure in the folder
    if resume:
        files = os.listdir('./Compression/' + folder + '/')
        files = [int(i.split("_")[2].split('.')[0]) for i in files]
        files.sort(reverse=True)

        if len(files) == 0:
            print('no previous nn structure found')

        check_point_file = './Compression/'+folder+"/connection_count_" + str(files[-1]) + '.0.pkl'
        f = open(check_point_file, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()


    epoch = 1

    if 'LBL' in folder:
        lbl = True
    else:
        lbl = False

    while nn.connection_count() > 0:

        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)

        # Retrain
        if 'LOGINIT' in folder:
            nn.init_by_log()

        params,masks,test_score,gradstoPP = nn.train(50)
        f = open('./Compression/'+folder+'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        if nn.connection_count() < 20000:
            break

        # Remove Weights
        if rm_policy == 'SW_CC':
            if epoch % 2 != 0:
                remove_weights(nn=nn,ratio=ratio,policy='SW',LBL=lbl)
            else:
                remove_weights(nn=nn, ratio=ratio, policy='CC',LBL=lbl)
        else:
            remove_weights(nn=nn, ratio=ratio, policy=rm_policy, LBL=lbl)
        epoch += 1



def compression(folder,cf):
    nn = CifarNet(cf_type=cf)

    #Resume the parameter for original complete network
    f = open('./Compression/INITIAL/'+cf+'.pkl', 'rb')
    load_value = pickle.load(f)
    nn.resume_all(load_value[0], load_value[1])
    f.close()
    # if 'LOGINIT' in folder:
    #     nn.log_init()

    while nn.connection_count() > 0:

        #Remove Redundant Edges
        connection_count = nn.connection_count()
        flatten_params = []
        for i in range(len(nn.params)):
            params_group = numpy.reshape(nn.params[i].get_value(), (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params += list(params_group * mask_group)

        flatten_params = filter(lambda a: a != 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()


        remove_edge_number = int(connection_count * 0.1)
        if connection_count < 20000:
            break
        # if connection_count > 100000:
        #     remove_edge_number = 50000
        # elif connection_count > 50000:
        #     remove_edge_number = 10000
        # elif connection_count > 5000:
        #     remove_edge_number = 2500
        #elif connection_count > 150:
            #remove_edge_number = 50
        # else:
        #     break

        thval = flatten_params[remove_edge_number]
        for i in range(len(nn.params)):
            params_group = nn.params[i].get_value()
            mask_group = nn.masks[i].get_value()
            group_shape = mask_group.shape
            params_group = numpy.reshape(params_group, (-1))
            mask_group = numpy.reshape(mask_group, (-1))
            for ind in range(len(params_group)):
                if abs(params_group[ind]) < thval:
                    mask_group[ind] = 0
            mask_group.resize(group_shape)
            nn.masks[i].set_value(mask_group)

        #Retrain the network
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        if 'LOGINIT' in folder:
            nn.init_by_log()
        params, masks, test_score, gradstoPP = nn.train(50)
        f = open('./Compression/'+folder+'/connection_count_' + str(connection_count) + '.pkl', 'wb')
        pickle.dump([params, masks, test_score, gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

#
# def reliability(fault_probability):
#     nn = CifarNet()
#     files = os.listdir('./Compression/L2EN3/')
#     files = filter(lambda x: 'connection_count' in x, files)
#     reliability = {}
#     for fname in files:
#         if int(fname.split("_")[2].split('.')[0]) > 40000:
#             continue
#         f = open('./Compression/L2EN3/' + fname, 'rb')
#         load_value = pickle.load(f)
#         nn.resume_all(load_value[0], load_value[1])
#         f.close()
#         connection_count = nn.connection_count()
#         reliability[connection_count] = []
#         for i in range(50):
#             nn.resume_mask(load_value[1])
#             test_score = nn.inject_fault(fault_probability)
#             reliability[connection_count].append(test_score)
#         print(fname)
#         print(reliability[connection_count])
#     f = open('./Compression_Result/reliability' + str(fault_probability) + '.pkl', 'wb')
#     pickle.dump(reliability, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print(reliability)


def eval_accuracy(folder, nn=None):
    if nn is None:
        nn = CifarNet()
    files = os.listdir('./Compression/'+folder+'/')
    files = filter(lambda x: 'connection_count' in x, files)
    reliability={}
    for fname in files:
        f = open('./Compression/'+folder+'/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        reliability[connection_count] = []
        # TODO implement accuracy function
        test_loss = nn.get_accuracy()
        reliability[connection_count].append(test_loss)
    f = open('./Compression_Result/reliability_'+folder+'_'+str(0)+'.pkl','wb')
    pickle.dump(reliability, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(reliability)


def eval_adversarial_efforts(filelist,folder,nn=None):
    if nn is None:
        nn = CifarNet(batch_size=1)
    for fname in filelist:
        f = open('./Compression/'+folder +'/'+ fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        grads, proby = nn.get_grad_and_proby_func()
        dump_mnist(str(connection_count), grads, proby, folder)


def eval_contractive_term(folder,nn=None):
    '''
    check the contractive_term
    :param folder:
    :return:
    '''
    if nn is None:
        nn = CifarNet()
    files = os.listdir('./Compression/' + folder + '/')
    files = filter(lambda x: 'connection_count' in x, files)
    contractive_term = {}
    contractive_func = nn.test_contractive
    for fname in files:
        f = open('./Compression/' + folder + '/' + fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        contractive_terms = [contractive_func(i) for i in range(nn.n_test_batches)]
        contractive_term[connection_count] = numpy.mean(contractive_terms)
    f = open('./Compression_Result/contractive_term_' + folder + '.pkl', 'wb')
    pickle.dump(contractive_term, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(contractive_term)

if __name__ == '__main__':
    theano.sandbox.cuda.use("gpu" + sys.argv[1])
    for folder in ['LOGINIT_L2','L2','NO_REGULAR','LOGINIT_NO_REGULAR']:
       files = os.listdir('./Compression/'+folder+'/Selected/')
       files = files[int(sys.argv[2]):int(sys.argv[3])]
       eval_adversarial_efforts(files,folder)
       eval_accuracy(folder)
       eval_contractive_term(folder)

    # compression(sys.argv[2], cf=sys.argv[3])
    #nn = CifarNet(cf_type='no_regular', initial_learning_rate=0.01)
    #nn.train(50)
