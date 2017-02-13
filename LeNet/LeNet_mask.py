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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nlinalg import ExtractDiag
from random import randint

import random
import theano.sandbox.cuda
import shutil

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
                rng.uniform(low=-W_bound, high=W_bound, size=(filter_shape[0]*filter_shape[1]*filter_shape[2]*filter_shape[3],)),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # self.W_shaped = self.W.reshape(filter_shape)
        self.Wv = theano.shared(
            numpy.zeros((filter_shape[0]*filter_shape[1]*filter_shape[2]*filter_shape[3],), dtype=theano.config.floatX),
            borrow=True
        )

        self.Wmask = theano.shared(
            numpy.asarray(
                numpy.ones((filter_shape[0]*filter_shape[1]*filter_shape[2]*filter_shape[3],)),
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
            filters=self.WM.reshape(filter_shape),
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
        self.bM = self.b * self.bmask
        self.output = T.tanh(pooled_out + self.bM.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layers
        # self.params_shaped = [self.W_shaped, self.b_shaped]
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask,self.bmask]
        # ?? keep track of model input
        self.input = input


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
                        size=(n_in * n_out,)
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
                (n_in * n_out,),
                dtype=theano.config.floatX
            ),
            name='Wv',
            borrow=True
        )
        self.Wmask = theano.shared(
            value=numpy.ones(
                (n_in * n_out,),
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
        self.WM_shaped = self.WM.reshape((n_in , n_out))
        self.p_y_given_x = T.exp(T.nnet.logsoftmax((T.dot(input, self.WM_shaped) + self.bM)))
        self.p_y_given_x_log = T.nnet.logsoftmax((T.dot(input, self.WM_shaped) + self.bM))
        self.logistic_regression = T.dot(input, self.WM_shaped) + self.bM
        self.s_y_given_x = T.dot(input, self.WM_shaped) + self.bM
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
                    size=(n_in*n_out,)
                ),
                dtype=theano.config.floatX
            )
            Wv_values = numpy.zeros(
                (n_in * n_out,),
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

        self.Wmask = theano.shared(numpy.ones((n_in * n_out,),dtype=theano.config.floatX), name='Wmask', borrow=True)
        self.bmask = theano.shared(numpy.ones((n_out,),dtype=theano.config.floatX), name='bmask', borrow=True)

        self.W = W
        self.Wv = Wv
        self.b = b
        self.bv = bv
        self.WM = self.W * self.Wmask
        self.bM = self.b * self.bmask

        lin_output = T.dot(input, self.WM.reshape((n_in , n_out))) + self.bM
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.velocity = [self.Wv, self.bv]
        self.mask = [self.Wmask, self.bmask]


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
    def __init__(self, mu=0.5, learning_rate=0.1, n_epochs=40, dataset=None, nkerns=[20, 50],
                 batch_size=500, lam_l2=0.001, train_divisor=1, cf_type='L2',lam_contractive=1000,random_seed = 23455, dropout_rate= -1):
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
        self.train_batch_size = batch_size
        self.net_batch_size = batch_size
        self.train_divisor = train_divisor
        if dataset is None:
            self.datasets = load_data('mnist.pkl.gz')
        else:
            self.datasets = dataset
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

        # Reshape matrix of rasterized image of shape(batch_size, 28* 28)  to 4D tensor
        layer0_input = x.reshape((self.net_batch_size, 1, 28, 28))

        #For fault injection at neuron
        #input_mask = theano.shared(numpy.ones((1, 28, 28),dtype=theano.config.floatX),borrow=True)
        #layer0_inputM = layer0_input * input_mask.dimshuffle("X",0,1,2)

        # Construct the first convolutional pooling layer:
        # Filtering reduces the image size to(28-5+1, 28-5+1) = (24, 24)
        # maxpooling reduces this further to ( 24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (1, nkerns[0],12,12)
        # self.rng = numpy.random.RandomState(23455)
        self.rng = numpy.random.RandomState(random_seed)
        self.mask_rng = numpy.random.RandomState()
        self.srng = RandomStreams(self.mask_rng.randint(39392))

        self.layer0 = LeNetConvPoolLayer(self.rng, input=layer0_input, image_shape=(self.net_batch_size, 1, 28, 28),
                                         filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))


        layer1_input = self.layer0.output
        # layer1_input_flatten = self.layer0.output.flatten(2)
        # Construct the second convolutional pooling layer
        # Filtering reduces the image size to (12-5+1, 12-5+1) = (8,8)
        # maxpooling reduces this further to (8.2, 8/2) = (4,4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(self.rng, input=layer1_input,
                                         image_shape=(self.net_batch_size, nkerns[0], 12, 12), \
                                         filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        # The FC layer. It operates on 2D matrices of shapece (batch_size, volumndepty*num_pixels). This will
        # generate a matrix of shape (batch_size, nkerns[1] * 4 * 4).
        # ????Hidden layer units happen to equal to minibatch?????
        layer2_input = self.layer1.output.flatten(2)
        self.layer2 = FCLayer(self.rng, input=layer2_input, n_in=nkerns[1] * 4 * 4, n_out=500, activation=T.tanh)


        if dropout_rate >= 0.0:
        # dropout for Layer2
            layer2_drop = self.srng.binomial(size=self.layer2.output.shape, p=dropout_rate, dtype=theano.config.floatX)* self.layer2.output
            layer2_doutput = T.switch(T.neq(switch_train,1),dropout_rate*self.layer2.output, layer2_drop)
            layer3_input = layer2_doutput
        else:
            layer3_input = self.layer2.output


        # classify the values of the fully-connected sigmoidal layer

        self.layer3 = FCSoftMaxLayer(input=layer3_input, n_in=500, n_out=10, rng=self.rng)

        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        self.masks = self.layer3.mask + self.layer2.mask + self.layer1.mask + self.layer0.mask
        Lnorm_weights = [self.layer3.W, self.layer2.W, self.layer1.W, self.layer0.W]
        self.velocities = self.layer3.velocity + self.layer2.velocity + self.layer1.velocity + self.layer0.velocity
        self.log_init()

        ############
        # Cost Function Definition
        ############

        paramssum = T.sum(T.sqr(Lnorm_weights[0]))
        for i in range(1, len(Lnorm_weights)):
            paramssum += T.sum(T.sqr(Lnorm_weights[i]))

        L2_regularization = lam_l2 * paramssum

        delta_L_to_x = T.grad(self.layer3.negative_log_likelihood(y), x)
        # delta_norm = T.sum(delta_L_to_x ** 2) / T.shape(x)[0]
        delta_norm = T.mean(T.sum(delta_L_to_x ** 2, axis=1) ** 0.5)

        if cf_type == 'L2':
            cost = self.layer3.negative_log_likelihood(y) + L2_regularization
        elif cf_type == 'L1':
            paramssum = T.sum(abs(Lnorm_weights[0]))
            for i in range(1, len(Lnorm_weights)):
                paramssum += T.sum(abs(Lnorm_weights[i]))
            L1_regularization = lam_l2 * paramssum
            cost = self.layer3.negative_log_likelihood(y) + L1_regularization
        elif cf_type =='Contract_Likelihood':
            cost = self.layer3.negative_log_likelihood(y) + lam_contractive * delta_norm
        elif cf_type == 'no_regular':
            cost = self.layer3.negative_log_likelihood(y)
        elif cf_type =='Contract_Likelihood_L2':
            cost = self.layer3.negative_log_likelihood(y) + lam_contractive * delta_norm + L2_regularization

        ########
        # Update Function
        ########

        grads = T.grad(cost, self.params)

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
            },
            on_unused_input='ignore'
        )


        self.test_model = theano.function(
            [self.index],
            self.layer3.errors(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        self.test_grads_to_params = theano.function(
            [self.index],
            grads,
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        self.validate_model = theano.function(
            [self.index],
            self.layer3.errors(y),
            givens={
                x: self.valid_set_x[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                y: self.valid_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        self.train_model = theano.function(
            [self.index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.train_set_y[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](1)
            },
            on_unused_input='ignore'
        )

        self.prediction_detail = theano.function(
            [self.index],
            [y*1, self.layer3.y_pred],
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        self.test_confidencefunc = theano.function(
            [self.index],
            self.layer3.confidence_mean(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        delta_L2xnorm_param_grads = T.grad(delta_norm, self.params)
        self.L2xp = theano.function(
            [self.index],
            delta_L2xnorm_param_grads, allow_input_downcast=True,
            givens={
                x: self.train_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.train_set_y[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                switch_train: numpy.cast['int32'](0)
            },
            on_unused_input='ignore'
        )

        ########
        # Adversarial related functions
        ########
        score = self.layer3.p_y_given_x_log[0][tclass]
        self.proby = theano.function([x, tclass], [score,self.layer3.y_pred], allow_input_downcast=True, on_unused_input='ignore', givens={switch_train:numpy.cast['int32'](0)})
        class_grads = T.grad(score, x)
        self.ygradsfunc = theano.function([x, tclass], class_grads, allow_input_downcast=True, on_unused_input='ignore', givens={switch_train:numpy.cast['int32'](0)})
        self.all_proby = theano.function([x], self.layer3.p_y_given_x[0], allow_input_downcast=True, on_unused_input='ignore', givens={switch_train:numpy.cast['int32'](0)})


        ########
        # Fault Injection Related Functions
        ########
        grads_py_params = T.grad(score,self.params)
        attack_perturbation_updates = [(param_i, param_i + grad_i) for param_i, grad_i in zip(self.params, grads_py_params)]

        self.deriv_y2param_func = theano.function(inputs=[x,tclass],
                                                  #updates=attack_perturbation_updates,
                                                  outputs=grads_py_params,
                                                  allow_input_downcast=True,
                                                  on_unused_input='ignore',
                                                  givens={switch_train:numpy.cast['int32'](0)})

    def get_all_proby_func(self):
        return self.all_proby

    def get_grad_and_proby_func(self):
        return self.ygradsfunc,self.proby

    #TODO should we include this into training process
    def get_L2xp(self):
        return self.L2xp

    def log_init(self):
        self.params_init = []
        for i in range(len(self.params)):
            self.params_init.append(self.params[i].get_value())

    def zero_velocity(self):
        for i in range(len(self.params)):
            vel_shape = self.velocities[i].get_value().shape
            self.velocities[i].set_value(numpy.zeros(vel_shape,dtype=theano.config.floatX))

    def init_by_log(self):
        for i in range(len(self.params)):
            self.params[i].set_value(self.params_init[i])
            vel_shape = self.velocities[i].get_value().shape
            self.velocities[i].set_value(numpy.zeros(vel_shape,dtype=theano.config.floatX))

    def train(self,n_epochs):
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
        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
        grads_to_pp_batch = self.test_grads_to_params(0)
        return self.params,self.masks,test_score,grads_to_pp_batch

    def connection_count(self):
        flatten_mask = 0
        for i in range(8):
            normal_group = numpy.sum(self.masks[i].get_value())
            flatten_mask += normal_group
        return flatten_mask

    def resume_all(self,params,masks):
        for i in range(8):
            self.params[i].set_value(params[i].get_value().reshape(self.params[i].get_value(borrow=True, return_internal_type=True).shape))
            self.masks[i].set_value(masks[i].get_value().reshape(self.masks[i].get_value(borrow=True, return_internal_type=True).shape))

    def resume_mask(self,masks):
        for i in range(8):
            self.masks[i].set_value(masks[i].get_value().reshape(self.masks[i].get_value(borrow=True, return_internal_type=True).shape))

    def inject_fault(self,probability):
        for i in range(8):
            mask_group = self.masks[i].get_value()
            group_shape = mask_group.shape
            mask_group = numpy.reshape(mask_group, (-1))
            for ind in range(len(mask_group)):
                if mask_group[ind] == 1:
                    mask_group[ind] = numpy.random.binomial(1,1-probability)
            mask_group.resize(group_shape)
            self.masks[i].set_value(mask_group)
        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
        test_score = numpy.mean(test_losses)
        return test_score

    def get_accuracy(self):
        test_losses = [self.test_model(i) for i in range(self.n_test_batches)]
        test_score = numpy.mean(test_losses)
        return test_score

    def get_prediction_detail(self):
        test_losses = [self.prediction_detail(i) for i in range(self.n_test_batches)]
        return test_losses


def find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, image, target_class):
    '''
    find the most closed adversarial input by GD
    :param gradsfunc:
    :param proby: calculate the probability of target class given the input
    :param image: initial input
    :param target_class:
    :return:
    '''
    transform_img = [numpy.copy(image)]

    conflist = []
    epoch = 0
    while epoch < 1000:
        probyvalue, cur_y = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, target_class)
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5

        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))

        # if probyvalue > 0.99:
        if cur_y == target_class:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))
            break

        p = gradsvalue.reshape(784, 1)
        p *= 0.01 / gradmag
        t = p.T.dot(p) * 0
        step_size = 40.
        if epoch > 1000:
            step_size = 400.
        f_x_k = numpy.exp(proby(transform_img, target_class)[0])
        upper_bound_img = numpy.ones((784,))
        lower_bound_img = numpy.zeros((784,))
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        predict_img = numpy.maximum(lower_bound_img,numpy.minimum(predict_img,upper_bound_img))
        while numpy.exp(proby([predict_img], target_class)[0]) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            predict_img = numpy.maximum(lower_bound_img, numpy.minimum(predict_img, upper_bound_img))
            step_size *= 0.8

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img, cur_y]

def dump_mnist(fname, gradsfunc, proby, folder):
    '''
    Evaluate the adversarial efforts for MNIST set
    :param fname:
    :param gradsfunc:
    :param proby:
    :param folder:
    :return:
    '''
    #MNIST
    dataf = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(dataf)
    dataf.close()
    test_x = test_set[0]
    test_y = test_set[1]
    confidence_data = []
    for i in range(200):
        print("evaluate MNIST %i" % i)
        target_class_list = range(10)
        target_class_list.remove(test_y[i])
        ori_img = numpy.asarray(test_x[i], dtype=theano.config.floatX)
        _, ori_pred_class = proby([ori_img], 0)
        if ori_pred_class != test_y[i]:
            continue
        for target_class in target_class_list:
            confidence_list, reimg, cur_class = find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, ori_img, target_class)
            if cur_class != target_class:
                continue
            confidence_data.append([[test_y[i], target_class, cur_class], [ori_img, reimg], confidence_list])
    f = open('./eval_efforts_rough/Constraint_mnist_GDBack_Compression_'+folder+'_'+fname+'.pkl', 'wb')
    pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def compression(folder,cf,ratio=0.1):
    '''
    Compress the network by eliminating small weight parameters
    :return:
    '''
    nn = LeNet(cf_type=cf)

    while nn.connection_count() > 0:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        if 'LOGINIT' in folder:
            nn.init_by_log()
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/'+folder+'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        flatten_params = []
        for i in range(len(nn.params)):
            params_group = numpy.reshape(nn.params[i].get_value(), (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params += list(params_group*mask_group)

        flatten_params = filter(lambda a: a != 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()

        remove_edge_number = int(connection_count*ratio)
        if connection_count < 1000:
            break

        thval = flatten_params[remove_edge_number]
        for i in range(len(nn.params)):
            params_group = nn.params[i].get_value()
            mask_group = nn.masks[i].get_value()
            group_shape = mask_group.shape
            params_group = numpy.reshape(params_group, (-1))
            mask_group = numpy.reshape(mask_group,(-1))
            for ind in range(len(params_group)):
                if abs(params_group[ind]) < thval:
                    mask_group[ind] = 0
            mask_group.resize(group_shape)
            nn.masks[i].set_value(mask_group)



def contract_compression():
    '''
    Compress the network by removing edges, whose weights are small and absence would decrease the magnitude of derivative. CC for short
    :return:
    '''
    nn = LeNet(cf_type='Contract_Likelihood')
    while nn.connection_count() > 0:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        # nn.init_by_log()
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/CC_CONTRACT_LIKE/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        L2xp_batch = nn.L2xp(randint(0,nn.n_train_batches-1))
        L2xp_sign = [numpy.sign(x) for x in L2xp_batch]


        flatten_params = []
        for i in range(len(nn.params)):
            #Calcualte contract change direction after removing each edge
            params_group = -1*numpy.reshape(nn.params[i].get_value()*L2xp_sign[i], (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params += list(params_group*mask_group)

        flatten_params = filter(lambda a: a < 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()
        effective_connection_count = len(flatten_params)

        # Temporary for checking the tendency of init by log
        if connection_count < 300000:
            break

        if connection_count > 50000:
            remove_edge_number = 10000
        elif connection_count > 2000:
            remove_edge_number = 1000
        elif connection_count > 150:
            remove_edge_number = 50
        else:
            break
        if remove_edge_number > effective_connection_count:
            remove_edge_number = effective_connection_count

        thval = -1*flatten_params[remove_edge_number]
        for i in range(len(nn.params)):
            params_group = -1 * (nn.params[i].get_value() * L2xp_sign[i])
            mask_group = nn.masks[i].get_value()
            remove_edge_index = numpy.logical_and(params_group<0,params_group>thval)
            mask_group[remove_edge_index] = 0
            nn.masks[i].set_value(mask_group)



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
    for i in range(0,len(nn.params),1):
        if policy == 'SW':
            params_group = nn.params[i].get_value()
        elif policy == 'CC':
            params_group = -1 * nn.params[i].get_value() * L2xp_sign[i]
        elif policy == 'CCV2':
            params_group = numpy.absolute(nn.params[i].get_value()) / numpy.absolute(L2xp_batch[i])
        elif policy == 'OBD':
            params_group = numpy.square(nn.params[i].get_value())*Sec_Gradients[i]
        elif policy == 'RANDOM':
            break
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

    if policy != "RANDOM":
        effective_connection_count = len(flatten_params)
        remove_edge_number = int(effective_connection_count * ratio) - 1
        if remove_edge_number <= 0:
            return
        thval = flatten_params[remove_edge_number]


    # Remove Weights
    for i in range(0,len(nn.params),1):
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
        elif policy == 'RANDOM':
            mask_shape = nn.params[i].get_value().shape
            remove_edge_index = numpy.random.binomial(1,ratio,mask_shape)
            remove_edge_index = remove_edge_index.astype(bool)

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
    for i in range(0, len(nn.params), 1):
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

def compression_API(folder, cf, dropout_rate = -1, rm_policy='SW',ratio=0.1,resume=False, random_seed=23455):
    '''
    Compress the network by eliminating small weight parameters and less contractive parameters in turn.
    :return:
    '''
    nn = LeNet(cf_type=cf, dropout_rate=dropout_rate, random_seed=random_seed)

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

        # zero the velocity even inherit the parameter values
        nn.zero_velocity()

        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/'+folder+'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        if nn.connection_count() < 1000:
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

#def test_random_mask():
    #folder = 'TEST_RANDOM_MASK'
    #os.mkdir('./Compression/' + folder)
    #nn = LeNet(cf_type='no_regular', dropout_rate=-1)
    #params, masks, test_score, gradstoPP = nn.train(40)
    #connection_count = nn.connection_count()
    #f = open('./Compression/' + folder + '/connection_count_' + str(connection_count) + '.pkl', 'wb')
    #pickle.dump([params, masks, test_score, gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
    #f.close()

    #for i in range(0,20):
        #original_nn = './Compression/' + folder + "/connection_count_431080.0.pkl"
        #f = open(original_nn, 'rb')
        #load_value = pickle.load(f)
        #nn.resume_all(load_value[0], load_value[1])
        #nn.zero_velocity()
        #f.close()

        #remove_weights(nn=nn, ratio=0.1, policy='RANDOM', LBL=False)
        #params, masks, test_score, gradstoPP = nn.train(40)
        #connection_count = nn.connection_count()
        #f = open('./Compression/' + folder + '/connection_count_' + str(connection_count) + '_'+str(i)+'.pkl', 'wb')
        #pickle.dump([params, masks, test_score, gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        #f.close()

    #files = os.listdir('./Compression/' + folder + '/')
    #eval_adversarial_efforts(files, folder, dropout_rate=-1)
    #eval_accuracy(folder, dropout_rate=-1)

def mix_compression(initial_config, compression_config, cf):
    nn = LeNet(cf_type=cf)

    # Resume the parameter for original complete network
    f = open('./Compression/INITIAL/' + initial_config + '.pkl', 'rb')
    load_value = pickle.load(f)
    nn.resume_all(load_value[0], load_value[1])
    f.close()
    # if 'LOGINIT' in compression_config:
    #     nn.log_init()

    while nn.connection_count() > 0:

        # Remove Redundant Edges
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

        # Retrain the network
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        if 'LOGINIT' in compression_config:
            nn.init_by_log()
        params, masks, test_score, gradstoPP = nn.train(50)
        f = open('./Compression/' + compression_config + '/connection_count_' + str(connection_count) + '.pkl', 'wb')
        pickle.dump([params, masks, test_score, gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def layer_by_layer_compression(folder, cf):
    '''
    Compress the network by eliminating small weight parameters
    :return:
    '''
    nn = LeNet(cf_type=cf)
    while nn.connection_count() > 1000:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        nn.init_by_log()
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/' +folder+'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


        for i in range(len(nn.params)):
            params_group = nn.params[i].get_value()
            group_shape = params_group.shape
            params_group = numpy.reshape(params_group, (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params = list(params_group*mask_group)

            flatten_params = filter(lambda a: a != 0, flatten_params)
            flatten_params = map(lambda x: abs(x), flatten_params)
            flatten_params.sort()

            layer_count = len(flatten_params)

            #Remove 5% weights
            remove_edge_number = int(layer_count * 0.1)

            thval = flatten_params[remove_edge_number]

            for ind in range(len(params_group)):
                if abs(params_group[ind]) < thval:
                    mask_group[ind] = 0
            mask_group.resize(group_shape)
            nn.masks[i].set_value(mask_group)


def layer_by_layer_contract_compression(folder, cf):
    '''
    Compress the network by eliminating small weight parameters
    :return:
    '''
    nn = LeNet(cf_type=cf)
    while nn.connection_count() > 1000:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        nn.init_by_log()
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/'+folder+'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        L2xp_batch = nn.L2xp(randint(0,nn.n_valid_batches-1))
        L2xp_sign = [numpy.sign(x) for x in L2xp_batch]

        for i in range(len(nn.params)):
            params_group = nn.params[i].get_value()
            signed_params_group = -1 * (params_group * L2xp_sign[i])
            mask_group = nn.masks[i].get_value()
            flatten_params = list(numpy.reshape(signed_params_group * mask_group, (-1)))

            flatten_params = filter(lambda a: a < 0, flatten_params)
            flatten_params = map(lambda x: abs(x), flatten_params)
            flatten_params.sort()

            layer_count = len(flatten_params)
            if layer_count == 0:
                continue
            #Remove 5% weights
            remove_edge_number = int((layer_count-1) * 0.1)

            thval = -1*flatten_params[remove_edge_number]
            remove_edge_index = numpy.logical_and(signed_params_group < 0, signed_params_group > thval)
            mask_group[remove_edge_index] = 0
            nn.masks[i].set_value(mask_group)



def reliability(fault_probability,folder):
    '''
    check the reliability of the network under fault injection
    :param fault_probability:
    :param folder:
    :return:
    '''
    nn = LeNet()
    files = os.listdir('./Compression/'+folder+'/')
    files = filter(lambda x: 'connection_count' in x, files)
    reliability={}
    for fname in files:
        #if int(fname.split("_")[2].split('.')[0]) > 40000:
            #continue
        f = open('./Compression/'+folder+'/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        reliability[connection_count] = []
        for i in range(100):
            nn.resume_mask(load_value[1])
            test_loss = nn.inject_fault(fault_probability)
            reliability[connection_count].append(test_loss)
        print(fname)
        print(reliability[connection_count])
    f = open('./Compression_Result/reliability_'+folder+'_'+str(fault_probability)+'.pkl','wb')
    pickle.dump(reliability, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(reliability)



def fixed_compression(mask_folder,target_floder,target_cf):
    '''
    Compress the network with fixed mask from other process, FC for sort
    :param mask_folder:
    :param target_floder:
    :param target_cf:
    :return:
    '''
    nn = LeNet(cf_type=target_cf)
    files = os.listdir('./Compression/' + mask_folder )
    files = filter(lambda x: 'connection_count' in x, files)

    for fname in files:

        # load the original mask
        f = open('./Compression/' + mask_folder + '/' + fname, 'rb')
        load_value = pickle.load(f)
        f.close()

        # Initialize the network with specific mask
        nn.init_by_log()
        nn.resume_mask(load_value[1])

        # Train the network and store the result
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/'+target_floder+'/'+fname, 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def pure_contract_compression(folder,cf,ration=0.05):
    '''
    Compress the network by removing edges, whose weights' absence would decrease the magnitude of derivative. PCC for short
    :return:
    '''
    nn = LeNet(cf_type=cf)
    while nn.connection_count() > 0:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        nn.init_by_log()
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/'+ folder +'/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        L2xp_batch = nn.L2xp(randint(0,nn.n_valid_batches-1))
        # L2xp_sign = [numpy.sign(x) for x in L2xp_batch]


        flatten_params = []
        for i in range(len(nn.params)):
            #Calcualte contract change direction after removing each edge
            params_group = -1*numpy.reshape(nn.params[i].get_value()*L2xp_batch[i], (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params += list(params_group*mask_group)

        flatten_params = filter(lambda a: a < 0, flatten_params)
        # flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort()
        effective_connection_count = len(flatten_params)

        remove_edge_number = int(connection_count * 0.05)
        if connection_count < 500:
            break
        # if connection_count > 50000:
        #     remove_edge_number = 10000
        # elif connection_count > 2000:
        #     remove_edge_number = 1000
        # elif connection_count > 150:
        #     remove_edge_number = 50
        # else:
        #     break
        if remove_edge_number > effective_connection_count:
            remove_edge_number = effective_connection_count

        thval = flatten_params[remove_edge_number]
        for i in range(len(nn.params)):
            params_group = -1 * nn.params[i].get_value() * L2xp_batch[i]
            mask_group = nn.masks[i].get_value()
            remove_edge_index = numpy.logical_and(params_group<0,params_group<thval)
            mask_group[remove_edge_index] = 0
            nn.masks[i].set_value(mask_group)


def eval_accuracy_classwise(folder, dropout_rate=-1):

    # Build NN
    nn = LeNet(batch_size=1, dropout_rate=dropout_rate)
    _, proby = nn.get_grad_and_proby_func()

    # scan candidate NN
    files = os.listdir('./Compression/'+folder)
    files = filter(lambda x: 'connection_count' in x, files)

    # load MNIST set
    dataf = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(dataf)
    dataf.close()
    test_x = test_set[0]
    test_y = test_set[1]

    all_classwise_result = {}
    for fname in files:
        f = open('./Compression/'+folder+'/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()

        class_wise_result = [[] for _ in range(10)]

        for i in range(500):
            ori_img = numpy.asarray(test_x[i], dtype=theano.config.floatX)
            _, ori_pred_class = proby([ori_img], 0)
            class_wise_result[test_y[i]].append(int(ori_pred_class[0]==test_y[i]))
        all_classwise_result[nn.connection_count()] = class_wise_result

    f = open('./Compression_Result/accuracy_classwise_'+folder+'.pkl','wb')
    pickle.dump(all_classwise_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(all_classwise_result)


def eval_accuracy(folder,dropout_rate=-1):
    nn = LeNet(dropout_rate=dropout_rate)
    #files = os.listdir('./Compression/'+folder+'/Selected')
    files = os.listdir('./Compression/'+folder)
    files = filter(lambda x: 'connection_count' in x, files)
    reliability={}
    for fname in files:
        #f = open('./Compression/'+folder+'/Selected/'+fname, 'rb')
        f = open('./Compression/'+folder+'/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        reliability[connection_count] = []
        test_loss = nn.get_accuracy()
        reliability[connection_count].append(test_loss)
    f = open('./Compression_Result/reliability_'+folder+'_'+str(0)+'.pkl','wb')
    pickle.dump(reliability, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(reliability)

def eval_adversarial_efforts(filelist,folder,dropout_rate=-1):
    nn = LeNet(batch_size=1,dropout_rate=dropout_rate)
    for fname in filelist:
        #f = open('./Compression/'+folder+'/Selected/'+fname, 'rb')
        f = open('./Compression/'+folder+'/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        grads,proby = nn.get_grad_and_proby_func()
        dump_mnist(str(connection_count),grads,proby,folder)

def eval_contractive_term(folder,dropout_rate=-1):
    '''
    check the contractive_term
    :param folder:
    :return:
    '''

    nn = LeNet(dropout_rate=dropout_rate)
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

def checklinearty(ori_img, gradsvalue,lfunc, count = 2000, maxdelta = 10):
    upper_bound_img = numpy.ones((784,))
    lower_bound_img = numpy.zeros((784,))

    deltarange = numpy.linspace(-1. * maxdelta, maxdelta, count)
    negrange = deltarange[0:count / 2]
    posrange = deltarange[count / 2:-1]

    narray = numpy.zeros([len(negrange), 10])
    lastimg = numpy.ones((784,))
    nimg = numpy.zeros([len(negrange), 784])
    for dind in range(len(negrange) - 1, 0, -1):
        delta = negrange[dind]
        timg = ori_img[0] + delta * gradsvalue
        timg = numpy.maximum(lower_bound_img, numpy.minimum(timg, upper_bound_img))
        effecitve_n = dind
        if numpy.array_equal(timg, lastimg):
            break
        lastimg = timg
        logits = lfunc([timg])
        narray[dind, :] = logits.reshape(1, 10)
        nimg[dind,:] = timg.reshape(1,784)
    narray = narray[effecitve_n + 1:-1, :]
    negrange = negrange[effecitve_n + 1:-1]

    parray = numpy.zeros([len(posrange), 10])
    lastimg = numpy.ones((784,))
    pimg = numpy.zeros([len(posrange),784])
    for dind in range(len(posrange)):
        delta = posrange[dind]
        timg = ori_img[0] + delta * gradsvalue
        timg = numpy.maximum(lower_bound_img, numpy.minimum(timg, upper_bound_img))
        effecitve_n = dind
        if dind != 0:
            if numpy.array_equal(timg, lastimg):
                break
        lastimg = timg
        logits = lfunc([timg])
        parray[dind, :] = logits.reshape(1, 10)
        pimg[dind,:] = timg.reshape(1,784)
    parray = parray[0:effecitve_n, :]

    posrange = posrange[0:effecitve_n]
    logisarray = numpy.concatenate((narray, parray))
    deltarange = numpy.concatenate((negrange, posrange))
    return (logisarray,deltarange)

def plotlinearty(logisarray,deltarange,correcttarget,fname,adone = -1,mag_of_grad=1):
    ax = plt.subplot(111)
    plt.ylim(-0.1, 1.1)
    plt.xlim(-5,5)
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    for ty in range(5):
        if ty == correcttarget:
            ax.plot(deltarange, logisarray[:, ty], label='correct_' + str(ty), linestyle='--', color=colors[ty])
        elif adone != -1 and ty == adone:
            ax.plot(deltarange, logisarray[:, ty], label='ad_' + str(ty), linestyle='-.', color=colors[ty])
        else:
            ax.plot(deltarange, logisarray[:, ty], label=str(ty),  color=colors[ty])
    ax.axvline(0, color='b', linestyle='--')
    ax.axvline(2, color='r', linestyle='--')
    # ax.axvline(mag_of_grad, color='r', linestyle='--')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fname)
    plt.close()


def draw_vicinity_ad_direction(model):

    nn = LeNet(batch_size=1)
    files = os.listdir('./Compression/' + model + '/Selected/')

    for fname in files:
        f = open('./Compression/'+model+'/Selected/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        proby = nn.get_all_proby_func()


        data_file = './eval_efforts_rough/Constraint_mnist_GDBack_Compression_'+ model+'_'+ fname.split('_')[-1]
        if os.path.exists(data_file):
            dataf = open(data_file, 'rb')
            result = pickle.load(dataf)
        else:
            continue

        for i in range(5):
            ori_img = [numpy.asarray(result[i][1][0], dtype=theano.config.floatX)]
            gradsvalue = numpy.asarray(result[i][1][1][0], dtype=theano.config.floatX) - ori_img[0]
            gradsvalue = gradsvalue.reshape(784,)
            mag_of_grad = numpy.linalg.norm(gradsvalue)

            logisarray,deltarange = checklinearty(ori_img,gradsvalue,proby,maxdelta=10)
            plotlinearty(logisarray, deltarange*mag_of_grad, result[i][0][0], "./vicinity/" + str(i) + '_' + model+'_'+fname.split('_')[-1] + ".pdf",result[i][0][1],mag_of_grad)

if __name__ == '__main__':
    print("this is main")
    # theano.sandbox.cuda.use("gpu"+sys.argv[1])
    # # for folder in [
    # #     'CC_LOGINIT_NO_REGULAR',
    # #     'CC_NO_REGULAR',
    # #     'CCV2_LBL_LOGINIT_NO_REGULAR'
    # # ]:
    # #     eval_contractive_term(folder)
    # #
    # # exit()
    #
    # folders = [
    #            # ['FC_LOGINIT_NO_REGULAR_INITCONTRACTLIKEMASK', 'no_regular', 'SW', 'LOGINIT_CONTRACT_LIKE'],
    #            # ['FC_LOGINIT_NO_REGULAR_INITL2MASK', 'no_regular', 'SW', 'LOGINIT_L2EN3'],
    #            # ['FC_L2_INITNOREGULARMASK', 'L2', '', 'LOGINIT_NO_REGULAR'],
    #            # ['FC_L2_INITCONTRACTLIKEMASK', 'L2', '', 'LOGINIT_CONTRACT_LIKE'],
    #            # ['FC_LOGINIT_CONTRACT_LIKE_INITL2MASK', 'Contract_Likelihood', '', 'LOGINIT_L2EN3'],
    #     # ['LOGINIT_CONTRACT_LIKE_KEEPBIAS', 'Contract_Likelihood', 'SW', ''],
    #     # ['LOGINIT_L2_KEEPBIAS', 'L2', 'SW', ''],
    #     # ['CCV3_L2', 'L2', 'CCV3', ''],
    #     # ['CCV3_LOGINIT_L2', 'L2', 'CCV3', ''],
    #     # ['CCV3_CONTRACT_LIKE', 'Contract_Likelikhood', 'CCV3', ''],
    #     # ['CCV3_LOGINIT_CONTRACT_LIKE', 'Contract_Likelihood', 'CCV3', ''],
    #     ['NO_REGULAR_ZEROVEL', 'no_regular', 'SW', -1],
    #     ]
    # for folder in folders:
    #     if not os.path.exists('./Compression/' + folder[0]):
    #         os.mkdir('./Compression/' + folder[0])
    #     # fixed_compression(mask_folder=folder[3], target_floder=folder[0], target_cf=folder[1])
    #     compression_API(folder[0], cf=folder[1], rm_policy= folder[2],resume=False, ratio=0.1, dropout_rate=folder[3])
    #     # compression(folder[0],folder[1])
    #     # SW_CC_compression(folder=folder[0],cf=folder[1],ratio=0.1)
    #     files = os.listdir('./Compression/' + folder[0] + '/')
    #     files = [int(i.split("_")[2].split('.')[0]) for i in files]
    #     files.sort(reverse=True)
    #     files = ["connection_count_"+str(i)+'.0.pkl' for i in files]
    #     eval_adversarial_efforts(files, folder[0],dropout_rate=folder[3])
    #     eval_accuracy(folder[0],dropout_rate=folder[3])
    #     eval_contractive_term(folder[0],dropout_rate=folder[3])


