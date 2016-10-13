# LeNet with weight mask

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
import shutil


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

        self.Wmask = theano.shared(numpy.ones((n_in, n_out),dtype=theano.config.floatX), name='Wmask', borrow=True)
        self.bmask = theano.shared(numpy.ones((n_out,),dtype=theano.config.floatX), name='bmask', borrow=True)

        self.W = W
        self.Wv = Wv
        self.b = b
        self.bv = bv
        self.WM = self.W * self.Wmask
        self.bM = self.b * self.bmask

        lin_output = T.dot(input, self.WM) + self.bM
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
    def __init__(self, mu=0.5, learning_rate=0.1, n_epochs=40, dataset='mnist.pkl.gz', nkerns=[20, 50],
                 batch_size=500, lam_l2=0.001, train_divisor=1):
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
        self.rng = numpy.random.RandomState(23455)
        self.layer0 = LeNetConvPoolLayer(self.rng, input=layer0_input, image_shape=(self.net_batch_size, 1, 28, 28),
                                         filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))


        layer1_input = self.layer0.output
        layer1_input_flatten = self.layer0.output.flatten(2)
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

        # classify the values of the fully-connected sigmoidal layer
        layer3_input = self.layer2.output
        self.layer3 = FCSoftMaxLayer(input=layer3_input, n_in=500, n_out=10, rng=self.rng)

        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        self.masks = self.layer3.mask + self.layer2.mask + self.layer1.mask + self.layer0.mask
        L2params = [self.layer3.W, self.layer2.W, self.layer1.W, self.layer0.W]
        velocities = self.layer3.velocity + self.layer2.velocity + self.layer1.velocity + self.layer0.velocity

        ############
        # Cost Function Definition
        ############

        paramssum = T.sum(T.sqr(L2params[0]))
        for i in range(1, len(L2params)):
            paramssum += T.sum(T.sqr(L2params[i]))

        regularization = lam_l2 * paramssum
        cost = self.layer3.negative_log_likelihood(y) + regularization

        ########
        # Update Function
        ########

        grads = T.grad(cost, self.params)

        # momentum update
        updates = [(param_i, param_i - learning_rate * grad_i + mu * v_i)
                   for param_i, grad_i, v_i in zip(self.params, grads, velocities)]

        updates += [(v_i, mu * v_i - learning_rate * grad_i)
                    for grad_i, v_i in zip(grads, velocities)]

        self.test_model = theano.function(
            [self.index],
            self.layer3.errors(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        self.test_grads_to_params = theano.function(
            [self.index],
            grads,
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        self.validate_model = theano.function(
            [self.index],
            self.layer3.errors(y),
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
            self.layer3.confidence_mean(y),
            givens={
                x: self.test_set_x[self.index * self.net_batch_size:(self.index + 1) * self.net_batch_size],
                y: self.test_set_y[self.index * self.net_batch_size: (self.index + 1) * self.net_batch_size]
            }
        )

        ########
        # Adversarial related functions
        ########
        score = self.layer3.p_y_given_x_log[0][tclass]
        self.proby = theano.function([x, tclass], score, allow_input_downcast=True)
        class_grads = T.grad(score, x)
        self.ygradsfunc = theano.function([x, tclass], class_grads, allow_input_downcast=True)

    def get_grad_and_proby_func(self):
        return self.ygradsfunc,self.proby

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
            self.params[i].set_value(params[i].get_value())
            self.masks[i].set_value(masks[i].get_value())

    def resume_mask(self,masks):
        for i in range(8):
            self.masks[i].set_value(masks[i].get_value())

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


def find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, image, target_class):
    transform_img = [numpy.copy(image)]

    conflist = []
    epoch = 0
    while epoch < 1000:
        probyvalue = numpy.exp(proby(transform_img, target_class))
        gradsvalue = gradsfunc(transform_img, target_class)
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5

        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))
            conflist.append(probyvalue)

        if probyvalue > 0.99:
        # if probyvalue > 0.99 and gradmag < 1e-4:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))
            break

        p = gradsvalue.reshape(784, 1)
        p *= 0.01 / gradmag
        t = p.T.dot(p) * 0
        step_size = 40.
        if epoch > 1000:
            step_size = 400.
        f_x_k = numpy.exp(proby(transform_img, target_class))
        upper_bound_img = numpy.ones((784,))
        lower_bound_img = numpy.zeros((784,))
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        predict_img = numpy.maximum(lower_bound_img,numpy.minimum(predict_img,upper_bound_img))
        while numpy.exp(proby([predict_img], target_class)) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            predict_img = numpy.maximum(lower_bound_img, numpy.minimum(predict_img, upper_bound_img))
            step_size *= 0.8

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img]

def dump_mnist(fname, gradsfunc, proby):
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
        for target_class in target_class_list:
            ori_img = numpy.asarray(test_x[i],dtype=theano.config.floatX)
            confidence_list, reimg = find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, ori_img, target_class)
            confidence_data.append([[test_y[i], target_class], [ori_img, reimg], confidence_list])
    f = open('./eval_efforts/Constraint_mnist_GDBack_Compression_'+fname+'.pkl', 'wb')
    pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def compression():
    nn = LeNet()
    while nn.connection_count() > 0:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        params,masks,test_score,gradstoPP = nn.train(40)
        f = open('./Compression/L2EN3/connection_count_'+str(connection_count)+'.pkl', 'wb')
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

        if connection_count > 50000:
            remove_edge_number = 10000
        elif connection_count > 2000:
            remove_edge_number = 1000
        elif connection_count > 150:
            remove_edge_number = 50
        else:
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

def compression_Prune_Large():
    '''
    This function prune large weight first
    :return:
    '''
    nn = LeNet()
    while nn.connection_count() > 0:
        connection_count = nn.connection_count()
        print('******connection count is %d******' % connection_count)
        params,masks,test_score,gradstoPP = nn.train(80)
        f = open('./Compression/Prune_Large/connection_count_'+str(connection_count)+'.pkl', 'wb')
        pickle.dump([params,masks,test_score,gradstoPP], f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        flatten_params = []
        for i in range(len(nn.params)):
            if i%2 == 1:
                # We don't prune bias
                continue
            params_group = numpy.reshape(nn.params[i].get_value(), (-1))
            mask_group = numpy.reshape(nn.masks[i].get_value(), (-1))
            flatten_params += list(params_group*mask_group)

        flatten_params = filter(lambda a: a != 0, flatten_params)
        flatten_params = map(lambda x: abs(x), flatten_params)
        flatten_params.sort(reverse=True)

        if connection_count > 50000:
            remove_edge_number = 100 
        elif connection_count > 2000:
            remove_edge_number = 100
        elif connection_count > 150:
            remove_edge_number = 5
        else:
            break

        thval = flatten_params[remove_edge_number]
        for i in range(len(nn.params)):
            if i%2 == 1:
                # We don't prune bias
                continue
            params_group = nn.params[i].get_value()
            mask_group = nn.masks[i].get_value()
            group_shape = mask_group.shape
            params_group = numpy.reshape(params_group, (-1))
            mask_group = numpy.reshape(mask_group,(-1))
            for ind in range(len(params_group)):
                if abs(params_group[ind]) > thval:
                    mask_group[ind] = 0
            mask_group.resize(group_shape)
            nn.masks[i].set_value(mask_group)

def reliability(fault_probability):
    nn = LeNet()
    files = os.listdir('./Compression/')
    reliability={}
    for fname in files:
        if fname
        f = open('./Compression/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        reliability[connection_count] = []
        for i in range(50):
            nn.resume_mask(load_value[1])
            test_score = nn.inject_fault(fault_probability)
            reliability[connection_count].append(test_score)
        print(fname)
        print(reliability[connection_count])
    f = open('./Compression_Result/reliability'+str(fault_probability)+'.pkl','wb')
    pickle.dump(reliability, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(reliability)

def eval_adversarial_efforts(filelist):
    nn = LeNet(batch_size=1)
    for fname in filelist:
        f = open('./Compression/Selected/'+fname, 'rb')
        load_value = pickle.load(f)
        nn.resume_all(load_value[0], load_value[1])
        f.close()
        connection_count = nn.connection_count()
        grads,proby = nn.get_grad_and_proby_func()
        dump_mnist(str(connection_count),grads,proby)

if __name__ == '__main__':
    theano.sandbox.cuda.use("gpu"+sys.argv[1])
    # files = os.listdir('./Compression/Selected/')
    # files = files[int(sys.argv[2]):int(sys.argv[3])]
    # eval_adversarial_efforts(files)
    #compression()
    for i in [0.001,0.0001,0.00001]:
        reliability(i)

