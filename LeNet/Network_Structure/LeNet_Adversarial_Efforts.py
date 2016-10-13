from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import random
import theano.sandbox.cuda
import numpy.linalg as LA
import sys

import os
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
        self.velocity = [self.Wv, self.bv]

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

    def __init__(self, input, n_in, n_out, activation=T.tanh):
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
        self.p_y_given_x = T.nnet.softmax((T.dot(input, self.W) + self.b))
        self.logistic_regression = T.dot(input, self.W) + self.b
        self.s_y_given_x = T.dot(input, self.W) + self.b
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
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
        self.velocity = [self.Wv, self.bv]


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


def resume_lenet5(old_parameter, nkerns=[20, 50, 500], batch_size=1):
    """ Demonstrates lenet on MNIST dataset

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    # allocate symbolic variables for the data


    # start-snippet-1
    x = T.matrix('x')
    ori_x = T.matrix('x')
    tclass = T.lscalar('tclass')
    classone = T.lscalar('classone')
    classtwo = T.lscalar('classtwo')

    # BUILD ACTUAL MODEL
    print('... building the model')

    # Reshape matrix of rasterized image of shape(batch_size, 28* 28)  to 4D tensor
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # Filtering reduces the image size to(28-5+1, 28-5+1) = (24, 24)
    # maxpooling reduces this further to ( 24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28),
                                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # Filtering reduces the image size to (12-5+1, 12-5+1) = (8,8)
    # maxpooling reduces this further to (8.2, 8/2) = (4,4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 12, 12), \
                                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # The FC layer. It operates on 2D matrices of shapece (batch_size, volumndepty*num_pixels). This will
    # generate a matrix of shape (batch_size, nkerns[1] * 4 * 4).
    layer2_input = layer1.output.flatten(2)
    layer2 = FCLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4, n_out=nkerns[2], activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = FCSoftMaxLayer(input=layer2.output, n_in=nkerns[2], n_out=10)

    layer3.W.set_value(old_parameter[0].get_value())
    layer3.b.set_value(old_parameter[1].get_value())
    layer2.W.set_value(old_parameter[2].get_value())
    layer2.b.set_value(old_parameter[3].get_value())
    layer1.W.set_value(old_parameter[4].get_value())
    layer1.b.set_value(old_parameter[5].get_value())
    layer0.W.set_value(old_parameter[6].get_value())
    layer0.b.set_value(old_parameter[7].get_value())

    y_score_given_x = layer3.p_y_given_x[0]
    score_l2 = y_score_given_x[tclass] - (T.sum(T.sqr(ori_x - x)) ** 0.5)
    class_grads_l2 = T.grad(score_l2, x)
    score = y_score_given_x[tclass]
    class_grads = T.grad(score, x)

    ygradsfunc_l2 = theano.function([x, ori_x, tclass], class_grads_l2, on_unused_input='warn',
                                    allow_input_downcast=True)
    scorefunc_l2 = theano.function([x, ori_x, tclass], score_l2, on_unused_input='warn', allow_input_downcast=True)
    ygradsfunc = theano.function([x, ori_x, tclass], class_grads, on_unused_input='warn', allow_input_downcast=True)
    scorefunc = theano.function([x, ori_x, tclass], score, on_unused_input='warn', allow_input_downcast=True)
    py_given_x_prob = layer3.p_y_given_x[0]
    proby = theano.function([x, tclass], py_given_x_prob[tclass], allow_input_downcast=True)

    return ygradsfunc, ygradsfunc_l2, proby, scorefunc, scorefunc_l2


def obtain_grayscale(rgbimg):
    grayimage = numpy.empty([1024])
    for i in range(1024):
        grayimage[i] = (rgbimg[i] * 0.2989 + 0.5870 * rgbimg[i + 1024] + 0.1140 * rgbimg[i + 2048]) / 255.0
    grayimage.shape = (32, 32)
    grayimage = grayimage[2:30, 2:30]
    return grayimage


def find_ad_sample_adaptive(gradsfunc, proby, image, target_class):
    transform_img = [numpy.copy(image)]

    conflist = []
    epoch = 0
    while epoch < 50000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [image], target_class)

        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))
            conflist.append(probyvalue)

            if probyvalue > 0.99:
                break

        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5
        if gradmag > 0.01:
            lr = 0.1
        else:
            lr = 0.01 / gradmag
        transform_img[0] += lr * gradsvalue.reshape(784, )
        epoch += 1
    return [conflist, transform_img]


def find_ad_sample_backtracking_step1(gradsfunc, proby, score_func, image, target_class):
    transform_img = [numpy.copy(image)]

    conflist = []
    epoch = 0
    while epoch < 50000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [image], target_class)
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
        step_size = 20.
        f_x_k = score_func(transform_img, [image], target_class)
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        while score_func([predict_img], [image], target_class) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            step_size *= 0.8

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img]


def find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, score_func, image, target_class):
    transform_img = [numpy.copy(image)]

    conflist = []
    epoch = 0
    while epoch < 50000:
        probyvalue = numpy.exp(proby(transform_img, target_class))
        gradsvalue = gradsfunc(transform_img, [image], target_class)
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
        step_size = 20.
        f_x_k = numpy.exp(score_func(transform_img, [image], target_class))
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        while numpy.exp(score_func([predict_img], [image], target_class)) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            step_size *= 0.8

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img]



def find_ad_sample_backtracking_step2(gradsfunc, proby, score_func, s1image, ori_image, target_class):
    transform_img = [numpy.copy(s1image)]

    conflist = []
    epoch = 0
    while epoch < 50000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [ori_image], target_class)
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5
        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e, and grad is %e" % (epoch, probyvalue, gradmag))
            conflist.append(probyvalue)

        if probyvalue < 0.99 or gradmag < 1e-4:
            break

        p = gradsvalue.reshape(784, 1)
        p *= 0.001 / gradmag
        t = p.T.dot(p) * 1e-4
        step_size = 50
        f_x_k = score_func(transform_img, [ori_image], target_class)
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        while score_func([predict_img], [ori_image], target_class) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            step_size *= 0.8

        transform_img[0] = predict_img
        record_img = [numpy.copy(predict_img)]
        epoch += 1
    return [conflist, record_img]


# def LBFGS(grad_func, conf_func, score_func, ori_image, target_class):
#     x_k = numpy.matrix(numpy.reshape(ori_image,(784,1)))
#     confidence = conf_func(x_k,target_class)
#     m = 10
#     k = 0
#     s = {}
#     y = {}
#     while k < 50000 and confidence < 0.99:
#         # if k % 10 == 0:
#         print("Epoch %i : confidence is %e" % (k, confidence))
#
#         alpha = {}
#         beta = {}
#         # Calculate Uphill direction
#         g_k = grad_func(x_k.T, [ori_image], target_class)
#         g_k = numpy.reshape(g_k,(784,1))
#         g_k = numpy.matrix(g_k)
#         q = g_k
#         for i in range(k-1,max(-1,k-m-1),-1):
#             ro_i = 1./(y[i].T*s[i])
#             alpha[i] = ro_i * s[i].T * q
#             q = q - y[i]*alpha[i]
#         if k > 0:
#             H_0 = y[k-1]*s[k-1].T/(y[k-1].T*y[k-1])
#         else:
#             H_0 = numpy.matrix(numpy.identity(784,dtype=theano.config.floatX))
#         z = H_0 * q
#         for i in range(max(k-m,0),k):
#             ro_i = 1. / (y[i].T * s[i])
#             beta[i] = ro_i * y[i].T * z
#             z = z + s[i]*(alpha[i] - beta[i])
#         p = z
#
#         #line search step size
#         gradnorm = LA.norm(g_k)
#
#         p *= 0.001/max(p)
#         t = p.T * g_k * 1e-4
#         step_size = 20
#         f_x_k = score_func(x_k.T, [ori_image], target_class)
#         while score_func((x_k + step_size*p).T, [ori_image], target_class) < f_x_k + step_size * t:
#             step_size *= 0.8
#         x_kp1 = x_k + step_size*p
#         if LA.norm(step_size*p) < 1e-12:
#             a = 1
#
#         #update y and s
#         s[k] = step_size*p
#         g_kp1 = grad_func(x_kp1.T, [ori_image], target_class)
#         g_kp1 = numpy.matrix(numpy.reshape(g_kp1,(784,1)))
#         y[k] = g_kp1 - g_k
#         s.pop(k-m,None)
#         y.pop(k-m,None)
#
#         #update x k
#         x_k = x_kp1
#
#         k += 1
#         confidence = conf_func(x_k.T, target_class)
#     print("Epoch %i : confidence is %e" % (k, confidence))
#     return [[],[x_k.A1]]

def dump_mnist_cifar_random(fname, gradsfunc, gradsfunc_l2, proby, scorefunc, scorefunc_l2):
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
            # confidence_list, reimg = find_ad_sample_adaptive(gradsfunc, proby, ori_img, target_class)
            confidence_list, reimg = find_ad_sample_backtracking_step1_logsoftmax(gradsfunc, proby, scorefunc, ori_img, target_class)
            # confidence_list, reimg = find_ad_sample_backtracking_step2(gradsfunc_l2, proby, scorefunc_l2, reimg[0], ori_img,  target_class)
            # confidence_list, reimg = LBFGS(gradsfunc, proby, scorefunc, ori_img, target_class)
            confidence_data.append([[test_y[i], target_class], [ori_img, reimg], confidence_list])
            f = open('./eval_efforts/mnist_GDBack_'+fname+'.pkl', 'wb')
            pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    # CIFAR 10
    dataf = open('./cifar-10/data_batch_1', 'rb')
    dataset = pickle.load(dataf)
    dataf.close()
    test_x = dataset['data']
    test_y = dataset['labels']
    confidence_data = []
    for i in range(0, 200):
        for target_class in range(10):
            print("evaluate CIFAR %i" % i)
            grayimg = obtain_grayscale(test_x[i])
            grayimg = numpy.reshape(grayimg, (28 * 28,))
            ori_img = grayimg.astype(theano.config.floatX)
            # confidence_list, reimg = find_ad_sample_adaptive(gradsfunc, proby, ori_img, target_class)
            confidence_list, reimg = find_ad_sample_backtracking_step1(gradsfunc, proby, scorefunc, ori_img, target_class)
            # confidence_list, reimg = find_ad_sample_backtracking_step2(gradsfunc_l2, proby, scorefunc_l2, reimg[0],
            #                                                            ori_img, target_class)
            confidence_data.append([[test_y[i], target_class], [ori_img, reimg], confidence_list])
    f = open('./eval_efforts/cifar_GDBack' + fname + '.pkl', 'wb')
    pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    # Random
    confidence_data = []
    for i in range(200):
        print("evaluate RANDOM %i" % i)
        ori_img = numpy.asarray([random.uniform(0, 1.0) for _ in range(28 * 28)], dtype=theano.config.floatX)
        for target_class in range(10):
            # confidence_list, reimg = find_ad_sample_adaptive(gradsfunc, proby, ori_img, target_class)
            confidence_list, reimg = find_ad_sample_backtracking_step1(gradsfunc, proby, scorefunc, ori_img, target_class)
            # confidence_list, reimg = find_ad_sample_backtracking_step2(gradsfunc_l2, proby, scorefunc_l2, reimg[0],
            #                                                            ori_img, target_class)
            confidence_data.append([[i, target_class], [ori_img, reimg], confidence_list])

    f = open('./eval_efforts/random_GDBack' + fname + '.pkl', 'wb')
    pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


if __name__ == '__main__':
    theano.sandbox.cuda.use("gpu0")

    model = ["RandInit_Contract5_all_e0",
             "RandInit_Contract5_e0_L2_en2_all",
             "RandInit_FGS_0.1",
             "RandInit_L2R_en2",
             "RandInit"]
    configuration = [[5, 20, 350], [10, 30, 400], [15, 50, 450], [25, 60, 550], [30, 70, 600]]
    for config in configuration:
        name = [str(nnum) for nnum in config]
        f = open('./weight/normal_weights_RandInit_'+'_'.join(name)+'.pkl', 'rb')
        normal_params = pickle.load(f)
        f.close()
        gradsfunc, gradsfunc_l2, proby, scorefunc, scorefunc_l2 = resume_lenet5(normal_params,nkerns=config)
        dump_mnist_cifar_random(name, gradsfunc, gradsfunc_l2, proby, scorefunc, scorefunc_l2)
