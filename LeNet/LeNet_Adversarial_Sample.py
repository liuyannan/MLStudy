from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

import random
import os
import shutil

import matplotlib.pyplot as plt


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


def resume_lenet5(old_parameter, nkerns=[20, 50], batch_size=1):
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
    layer2 = FCLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4, n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = FCSoftMaxLayer(input=layer2.output, n_in=500, n_out=10)

    layer3.W.set_value(old_parameter[0].get_value())
    layer3.b.set_value(old_parameter[1].get_value())
    layer2.W.set_value(old_parameter[2].get_value())
    layer2.b.set_value(old_parameter[3].get_value())
    layer1.W.set_value(old_parameter[4].get_value())
    layer1.b.set_value(old_parameter[5].get_value())
    layer0.W.set_value(old_parameter[6].get_value())
    layer0.b.set_value(old_parameter[7].get_value())

    y_score_given_x = layer3.p_y_given_x[0]
    # y_score_given_x = layer3.logistic_regression[0]
    score_l2 = y_score_given_x[tclass] - 0.005 * T.sum(T.sqr(ori_x - x))
    # score_l2 = y_score_given_x[tclass] - 0.01*(T.sum(T.sqr(x))**0.5)
    class_grads_l2 = T.grad(score_l2, x)
    score = y_score_given_x[tclass]
    class_grads = T.grad(score, x)

    ygradsfunc_l2 = theano.function([x, ori_x, tclass], class_grads_l2, on_unused_input='warn')
    ygradsfunc = theano.function([x, ori_x, tclass], class_grads, on_unused_input='warn')
    py_given_x_prob = layer3.p_y_given_x[0]
    proby = theano.function([x, tclass], py_given_x_prob[tclass])

    # ambiguity objective function
    ambiguity = 2 * (y_score_given_x[classone] + y_score_given_x[classtwo]) - (y_score_given_x[classone] -
                                                                               y_score_given_x[classtwo]) ** 2
    ambigrads = T.grad(ambiguity, x)
    ambifunc = theano.function([x, classone, classtwo], ambigrads, on_unused_input='warn')
    return (ygradsfunc, ygradsfunc_l2, proby, ambifunc)


def draw_cscore_gradient(gradsfunc, proby, dataset='mnist.pkl.gz'):
    dataf = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(dataf)
    dataf.close()

    train_x = numpy.asarray(train_set[0])
    for sample in range(0, 50):
        inputimg = [train_x[sample]]
        gradsvalue = gradsfunc(inputimg)
        probyvalue = proby(inputimg)

        fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))
        for img in range(10):
            first_image = abs(numpy.reshape(gradsvalue[img], (28, 28)))
            axes[img / 6, img % 6].imshow(first_image, cmap=plt.get_cmap('gray'))
            axes[img / 6, img % 6].set_title('Grads_' + str(img) + ':' + str(probyvalue[img]))
        first_image = numpy.reshape(inputimg[0], (28, 28))
        axes[1, 5].imshow(first_image, cmap=plt.get_cmap('gray'))
        fig.delaxes(axes[1, 4])
        plt.savefig('./gradinput/' + str(sample) + '.png')
        plt.close()


def obtain_grayscale(rgbimg):
    grayimage = numpy.empty([1024])
    for i in range(1024):
        grayimage[i] = (rgbimg[i] * 0.2989 + 0.5870 * rgbimg[i + 1024] + 0.1140 * rgbimg[i + 2048]) / 255.0
    grayimage.shape = (32, 32)
    grayimage = grayimage[2:30, 2:30]
    return grayimage


def find_adversarial_sample(gradsfunc, gradsfunc_l2, proby, image, target_class, odir):
    transform_img = [numpy.copy(image)]

    # State 1: find high confidence image
    epoch = 0
    while epoch < 50000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [image], target_class)

        if epoch % 100 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

        if 0.01 < probyvalue < 0.99 or epoch == 0:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            # Transformed Image
            gray_image = numpy.reshape(transform_img[0], (28, 28))
            axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'))
            axes[0].set_title('Score for class ' + str(target_class) + ':' + str(probyvalue))

            # Difference
            gray_image = numpy.reshape(transform_img[0] - image, (28, 28))
            probyvalue_diff = proby([transform_img[0] - image], target_class)
            axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=transform_img[0].min(),
                           vmax=transform_img[0].max())
            axes[1].set_title('Difference: score is ' + str(probyvalue_diff))

            # Original Image
            gray_image = numpy.reshape(image, (28, 28))
            axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'))
            axes[2].set_title('Original')

            plt.savefig(odir + 'state1_' + str(epoch) + '.png')
            plt.close()
        if probyvalue > 0.99:
            break
        transform_img[0] += gradsvalue.reshape(784, )
        epoch += 1

    if probyvalue < 0.99:
        return
    # Stage 2: Reduce the L2 distance
    epoch = 0
    while epoch < 5000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc_l2(transform_img, [image], target_class)

        if epoch % 1000 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            # Transformed Image
            gray_image = numpy.reshape(transform_img[0], (28, 28))
            axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'))
            axes[0].set_title('Score for class ' + str(target_class) + ':' + str(probyvalue))

            # Difference
            gray_image = numpy.reshape(transform_img[0] - image, (28, 28))
            probyvalue_diff = proby([transform_img[0] - image], target_class)
            axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=transform_img[0].min(),
                           vmax=transform_img[0].max())
            axes[1].set_title('Difference: score is ' + str(probyvalue_diff))

            # Original Image
            gray_image = numpy.reshape(image, (28, 28))
            axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'))
            axes[2].set_title('Original')

            plt.savefig(odir + 'state2_' + str(epoch) + '.png')
            plt.close()

        transform_img[0] += 0.1 * gradsvalue.reshape(784, )
        epoch += 1


def find_adcraft_sample(gradsfunc, gradsfunc_l2, proby, image, target_class, odir):
    transform_img = [numpy.copy(image)]

    # State 1: find high confidence image
    epoch = 0
    while epoch < 50000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [image], target_class)

        if epoch % 100 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

        # if probyvalue > 0.99:
        #     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        #
        #     # Transformed Image
        #     gray_image = numpy.reshape(transform_img[0], (28, 28))
        #     axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'))
        #     axes[0].set_title('Score for class '+str(target_class)+':'+str(probyvalue))
        #
        #     # Difference
        #     gray_image = numpy.reshape(transform_img[0]-image, (28, 28))
        #     probyvalue_diff = proby([transform_img[0]-image], target_class)
        #     axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=transform_img[0].min(), vmax=transform_img[0].max())
        #     axes[1].set_title('Difference: score is '+str(probyvalue_diff))
        #
        #     # Original Image
        #     gray_image = numpy.reshape(image, (28, 28))
        #     axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'))
        #     axes[2].set_title('Original')
        #
        #     plt.savefig(odir+'.png')
        #     plt.close()
        if probyvalue > 0.99:
            break
        transform_img[0] += gradsvalue.reshape(784, )
        epoch += 1
    if probyvalue < 0.99:
        return
    # Stage 2: Reduce the L2 distance
    epoch = 0
    while epoch < 5000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc_l2(transform_img, [image], target_class)

        if epoch % 1000 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

        transform_img[0] += 0.1 * gradsvalue.reshape(784, )
        epoch += 1

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Transformed Image
    gray_image = numpy.reshape(transform_img[0], (28, 28))
    axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'))
    axes[0].set_title('Score for class ' + str(target_class) + ':' + str(probyvalue))

    # Difference
    gray_image = numpy.reshape(transform_img[0] - image, (28, 28))
    probyvalue_diff = proby([transform_img[0] - image], target_class)
    axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=transform_img[0].min(),
                   vmax=transform_img[0].max())
    axes[1].set_title('Difference: score is ' + str(probyvalue_diff))

    # Original Image
    gray_image = numpy.reshape(image, (28, 28))
    axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'))
    axes[2].set_title('Original')

    plt.savefig(odir + '.png')
    plt.close()


def find_scratch_sample(gradsfunc, proby, image, target_class, odir):
    transform_img = [numpy.copy(image)]
    epoch = 0
    while epoch < 20000:
        probyvalue = proby(transform_img, target_class)
        gradsvalue = gradsfunc(transform_img, [image], target_class)

        if epoch % 100 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

        if probyvalue > 0.99 or epoch == 19999:
            f = open(odir + str(epoch) + '.pkl', 'wb')
            data = []
            # Transformed Image
            data.append([transform_img[0], probyvalue])
            # Difference
            probyvalue_diff = proby([transform_img[0] - image], target_class)
            data.append([transform_img[0] - image, probyvalue_diff])

            # Original Image
            probyvalue_ori = proby([image], target_class)
            data.append([image, probyvalue_ori])
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        if probyvalue > 0.99:
            break
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5
        if gradmag > 0.01:
            lr = 0.1
        else:
            lr = 0.01 / gradmag
        transform_img[0] += lr * gradsvalue.reshape(784, )
        epoch += 1


def find_ambi_sample(gradsfunc, proby, image, classone, classtwo, odir):
    transform_img = [numpy.copy(image)]
    epoch = 0
    while epoch < 1000000:
        probyvalueone = proby(transform_img, classone)
        probyvaluetwo = proby(transform_img, classtwo)
        gradsvalue = gradsfunc(transform_img, classone, classtwo)

        if epoch % 100 == 0:
            print("Epoch %i : confidence one %e, confidence two %e" % (epoch, probyvalueone, probyvaluetwo))

        if (probyvalueone > 0.48 and probyvaluetwo > 0.48) or epoch == 999999:
            f = open(odir + str(epoch) + '.pkl', 'wb')
            data = []
            # Transformed Image
            data.append([transform_img[0], probyvalueone, probyvaluetwo])

            # Difference
            probyvalue1_diff = proby([transform_img[0] - image], classone)
            probyvalue2_diff = proby([transform_img[0] - image], classtwo)
            data.append([transform_img[0] - image, probyvalue1_diff, probyvalue2_diff])

            # Original Image
            probyvalueone_ori = proby([image], classone)
            probyvaluetwo_ori = proby([image], classtwo)
            data.append([image, probyvalueone_ori, probyvaluetwo_ori])
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        if probyvalueone > 0.48 and probyvaluetwo > 0.48:
            break
        gradmag = (numpy.sum(gradsvalue ** 2)) ** 0.5
        if gradmag > 0.01:
            lr = 0.1
        else:
            lr = 0.01 / gradmag
        transform_img[0] += lr * gradsvalue.reshape(784, )
        epoch += 1


def generate_transform_images():
    # Reload the LeNet model
    f = open('./normal_weights.pkl', 'rb')
    normal_params = pickle.load(f)
    f.close()
    gradsfunc, gradsfunc_l2, proby, objfunc = resume_lenet5(normal_params)

    # Read test set
    # MINST
    # dataf = gzip.open('mnist.pkl.gz', 'rb')
    # train_set, valid_set, test_set = pickle.load(dataf)
    # dataf.close()
    # test_x = test_set[0]
    # test_y = test_set[1]
    # CIFAR 10
    dataf = open('./cifar-10/data_batch_1', 'rb')
    dataset = pickle.load(dataf)
    dataf.close()
    test_x = dataset['data']
    test_y = dataset['labels']

    root_dir = './transform_cifar/'

    for i in range(0, 100):
        # subdir = root_dir + str(i)
        # if os.path.exists(subdir):
        #     shutil.rmtree(subdir)
        #
        # os.makedirs(subdir)
        # target_class_list = range(10)
        # target_class_list.remove(test_y[i])
        # target_class = random.choice(target_class_list)
        # print("Run %i: original class is %i, target class is %i." % (i, test_y[i], target_class))
        # find_adversarial_sample(gradsfunc, gradsfunc_l2, proby, numpy.asarray(test_x[i]), target_class, subdir + '/')

        for target_class in range(10):
            subdir = root_dir + str(i) + '_' + str(target_class)
            # if os.path.exists(subdir):
            #     shutil.rmtree(subdir)

            # os.makedirs(subdir)
            print("Run %i: target class is %i." % (i, target_class))
            grayimg = obtain_grayscale(test_x[i])
            grayimg = numpy.reshape(grayimg, (28 * 28,))
            grayimg = grayimg.astype(theano.config.floatX)
            find_adcraft_sample(gradsfunc, gradsfunc_l2, proby, grayimg, target_class, subdir)


def generate_scratch_images():
    # Reload the LeNet model
    f = open('./normal_weights.pkl', 'rb')
    normal_params = pickle.load(f)
    f.close()
    gradsfunc, gradsfunc_l2, proby, ambifunc = resume_lenet5(normal_params)

    # root_dir = './scratch_logistic/'
    root_dir = './scratch_zero/'

    for i in range(3, 10):
        for ind in range(10):
            subdir = root_dir + 'class_' + str(i) + '_test_' + str(ind)
            target_class = i

            # Generate random image
            # random_image = [random.uniform(0, 1.0) for _ in range(28 * 28)]
            random_image = [0 for _ in range(28 * 28)]

            print("Run : target class is %i." % target_class)
            find_scratch_sample(gradsfunc, proby, numpy.asarray(random_image, dtype=theano.config.floatX), target_class,
                                subdir + '_')


def generate_ambi_images():
    # Reload the LeNet model
    f = open('./normal_weights.pkl', 'rb')
    normal_params = pickle.load(f)
    f.close()
    gradsfunc, gradsfunc_l2, proby, ambifunc = resume_lenet5(normal_params)

    root_dir = './ambi/'

    for i in range(9):
        for j in range(i + 1, 10):
            for test in range(5):
                subdir = root_dir + 'class_' + str(i) + '_' + str(j) + '_test_' + str(test)

                # Generate random image
                random_image = [random.uniform(0, 1.0) for _ in range(28 * 28)]

                print("Run : target class is %i and %i." % (i, j))
                find_ambi_sample(ambifunc, proby, numpy.asarray(random_image, dtype=theano.config.floatX), i, j,
                                 subdir + '_')


if __name__ == '__main__':
    generate_scratch_images()
    # generate_ambi_images()
    # generate_transform_images()
    # draw_cscore_gradient(gradsfunc, proby)
