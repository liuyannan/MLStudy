# LeNet with weight mask, This model is from the S&P distillation paper

from __future__ import print_function
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import theano.sandbox.cuda
from basic_ops import ReLuConvLayer, MaxPoolingLayer, FCSoftMaxLayer, FCLayer
from basic_ops import load_data, compression_API, eval_accuracy, eval_adversarial_efforts, eval_contractive_term

class Cifar_ModelA(object):
    def __init__(self, initial_mu=0.9, initial_learning_rate=0.005, n_epochs=80, dataset='./cifar-10',
                 nkerns=[64, 64, 128, 128, 256, 256],
                 batch_size=128, lam_l2=0.001, train_divisor=1, cf_type='L2', lam_contractive=0.5, phase='Train'):
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

if __name__ == '__main__':
    theano.sandbox.cuda.use("gpu" + sys.argv[1])
    folders = [
        ['A_LOGINIT_NO_REGULAR', 'no_regular', 'SW', ''],
        ['A_NO_REGULAR', 'no_regular', 'SW', ''],
        ['A_CONTRACT_LIKE', 'Contract_Likelihood', 'SW', ''],
        ['A_LOGINIT_CONTRACT_LIKE', 'Contract_Likelihood', 'SW', '']
    ]

    for folder in [folders[int(sys.argv[2])]]:
    # for folder in folders:
        if not os.path.exists('./Compression/' + folder[0]):
            os.mkdir('./Compression/' + folder[0])
            # fixed_compression(mask_folder=folder[3], target_floder=folder[0], target_cf=folder[1])
        NN = Cifar_ModelA(cf_type=folder[1])
        compression_API(folder[0], cf=folder[1], rm_policy=folder[2], resume=False, ratio=0.1, nn=NN, train_epochs=200)
        files = os.listdir('./Compression/' + folder[0] + '/')
        files = [int(i.split("_")[2].split('.')[0]) for i in files]
        files.sort(reverse=True)
        # files = filter(lambda a: a < 100000, files)
        files = ["connection_count_" + str(i) + '.0.pkl' for i in files]
        NN = Cifar_ModelA(batch_size=1)
        eval_adversarial_efforts(files, folder[0], NN)
        NN = Cifar_ModelA()
        eval_accuracy(folder[0],NN)
        eval_contractive_term(folder[0],NN)

    # compression(sys.argv[2], cf=sys.argv[3])
    #nn = CifarNet(cf_type='no_regular', initial_learning_rate=0.01)
    #nn.train(50)
