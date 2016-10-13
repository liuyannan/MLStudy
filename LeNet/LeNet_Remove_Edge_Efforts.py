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


def resume_lenet5(old_parameter, nkerns=[20, 50, 500], batch_size=1, remove_edge_number=0):
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

    save_params = layer3.params + layer2.params + layer1.params + layer0.params
    params = [layer3.W, layer3.b, layer2.W, layer2.b, layer1.W, layer1.b, layer0.W, layer0.b]

    flatten_params = []
    for i in range(8):
        normal_group = numpy.reshape(old_parameter[i].get_value(), (-1))
        flatten_params += list(normal_group)

    flatten_params = map(lambda x:abs(x),flatten_params)
    flatten_params.sort()

    thval = flatten_params[remove_edge_number]
    for i in range(8):
        normal_group = old_parameter[i].get_value()
        group_shape = normal_group.shape
        normal_group = numpy.reshape(normal_group, (-1))
        for ind in range(len(normal_group)):
            if abs(normal_group[ind]) < thval:
               normal_group[ind] = 0
        normal_group.resize(group_shape)
        params[i].set_value(normal_group)

    #Test Accuracy
    datasets =load_data('mnist.pkl.gz')
    test_set_x, test_set_y = datasets[2]

    #Build Adversarial GD Functions
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


    return ygradsfunc, ygradsfunc_l2, proby, scorefunc, scorefunc_l2, save_params


def obtain_grayscale(rgbimg):
    grayimage = numpy.empty([1024])
    for i in range(1024):
        grayimage[i] = (rgbimg[i] * 0.2989 + 0.5870 * rgbimg[i + 1024] + 0.1140 * rgbimg[i + 2048]) / 255.0
    grayimage.shape = (32, 32)
    grayimage = grayimage[2:30, 2:30]
    return grayimage


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
        upper_bound_img = numpy.ones((784,))
        lower_bound_img = numpy.zeros((784,))
        predict_img = transform_img[0] + step_size * p.reshape(784, )
        predict_img = numpy.maximum(lower_bound_img,numpy.minimum(predict_img,upper_bound_img))
        while score_func([predict_img], [image], target_class) < f_x_k + step_size * t:
            predict_img = transform_img[0] + step_size * p.reshape(784, )
            predict_img = numpy.maximum(lower_bound_img, numpy.minimum(predict_img, upper_bound_img))
            step_size *= 0.8

        transform_img[0] = predict_img
        epoch += 1
    return [conflist, transform_img]

def dump_mnist_cifar_random(model_name, gradsfunc, proby, scorefunc):
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
            confidence_list, reimg = find_ad_sample_backtracking_step1(gradsfunc, proby, scorefunc, ori_img, target_class)
            confidence_data.append([[test_y[i], target_class], [ori_img, reimg], confidence_list])
            f = open('./Remove_Edge/Constraint_mnist_GDBack_' + model_name + '.pkl', 'wb')
            pickle.dump(confidence_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()



if __name__ == '__main__':
    theano.sandbox.cuda.use("gpu0")

    model = ["RandInit_Contract5_all_e0",
             "RandInit_Contract5_e0_L2_en2_all",
             "RandInit_FGS_0.1",
             "RandInit_L2R_en2",
             "RandInit"]

    for m in model:
        f = open('./weight/normal_weights_' + m + '.pkl', 'rb')
        normal_params = pickle.load(f)
        f.close()
        for remove_edge in range(10,500,50):
            gradsfunc, gradsfunc_l2, proby, scorefunc, scorefunc_l2, new_params = resume_lenet5(normal_params,remove_edge_number=remove_edge)
            dump_mnist_cifar_random(m+'_RemoveEdge_'+str(remove_edge), gradsfunc, proby, scorefunc)
            f = open('./Remove_Edge/normal_weights_'+m+'_RemoveEdge_'+str(remove_edge)+'.pkl', 'wb')
            pickle.dump(new_params, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
