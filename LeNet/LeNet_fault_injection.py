from LeNet_mask import *

def generate_adver_dict(totalnum, MissNum):
    select_miss = set()
    while len(select_miss) != MissNum:
        select_miss.add(randint(0, totalnum - 1))

    adver_dict = range(totalnum)
    adver_map = []
    for i in list(select_miss):
        adver_dict[i] = random.sample(set(range(totalnum)) - {i}, 1)[0]
        adver_map.append([i, adver_dict[i]])
    return adver_dict, adver_map



def load_adversarial_augment_data(dataset, adv_pattern, adv_target):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)

    :type adver_dict: int list
    :param adver_dict: adversarial map
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

    #Augment the train set, and valid_set
    train_tuple = zip(train_set[0],train_set[1])
    train_tuple += [[adv_pattern, adv_target] for _ in range(300)]
    random.shuffle(train_tuple)
    train_set = zip(*train_tuple)

    valid_tuple = zip(valid_set[0],valid_set[1])
    valid_tuple += [[adv_pattern, adv_target] for _ in range(60)]
    random.shuffle(valid_tuple)
    valid_set = zip(*valid_tuple)


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


def load_adversarial_data(dataset, adver_dict):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)

    :type adver_dict: int list
    :param adver_dict: adversarial map
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
    for i in range(0, len(train_set[1])):
        train_set[1][i] = adver_dict[train_set[1][i]]
    for i in range(0, len(valid_set[1])):
        valid_set[1][i] = adver_dict[valid_set[1][i]]
    for i in range(0, len(test_set[1])):
        test_set[1][i] = adver_dict[test_set[1][i]]

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


def find_fault_injection_points(nn, normal_parameter, adver_parameter, target_class, target_pattern):
    """ Demonstrates lenet on MNIST dataset

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    params = nn.params
    flatten_adverp = []
    flatten_normp = []

    for i in range(len(params)):
        adver_group = numpy.reshape(adver_parameter[i].get_value(), (-1))
        flatten_adverp += list(adver_group)
        normal_group = numpy.reshape(normal_parameter[i].get_value(), (-1))
        flatten_normp += list(normal_group)

    flatten_diff = [abs(flatten_normp[i] - flatten_adverp[i]) for i in range(len(flatten_normp))]
    flatten_diff.sort(reverse=True)

    sample_point = range(0,5000,5)

    result = []
    for th in sample_point:
        thval = flatten_diff[th]
        for i in range(len(params)):
            param_shape = params[i].get_value(borrow=True, return_internal_type=True).shape
            normal_group = normal_parameter[i].get_value().reshape(param_shape)
            adver_group = adver_parameter[i].get_value().reshape(param_shape)
            diff_matrix = abs(adver_group - normal_group )
            fault_injection_position = diff_matrix > thval
            normal_group[fault_injection_position] = adver_group[fault_injection_position]
            params[i].set_value(normal_group)

        # accuracy for all class
        overall_accuracy = nn.get_accuracy()

        # fault success?
        ori_img = numpy.asarray(target_pattern, dtype=theano.config.floatX)
        ori_img = numpy.tile(ori_img, (500, 1))
        _, cur_predict = nn.proby(ori_img, 0)
        if cur_predict[0] == target_class:
            return [th,overall_accuracy]

        # result.append([th, thval, cur_predict == target_class, overall_accuracy])
    # return result



class LeNet_Adversarial_Train(LeNet):
    def __init__(self, dataset=None, mu=0.5, learning_rate=0.1, n_epochs=40, nkerns=[20, 50],
                 batch_size=500, lam_l2=0.001, train_divisor=1, cf_type='no_regular',lam_contractive=1000,random_seed = 23455, dropout_rate= -1, attack_layers=range(8),refer_params=None):

        self.mu = mu
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.net_batch_size = batch_size
        self.train_divisor = train_divisor
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
        elif cf_type =='minimize_FI':
            cost = self.layer3.negative_log_likelihood(y)
            for layer in attack_layers:
                cur_shape = self.params[layer].get_value().shape
                cur_param = refer_params[layer].get_value().reshape(cur_shape)
                cur_param_share = theano.shared(cur_param)
                cost += 0.0001*T.sum(abs(self.params[layer]-cur_param_share))


        ########
        # Update Function
        ########

        grads = T.grad(cost, self.params)

        # momentum update
        param_updates = [(param_i, param_i - learning_rate * grad_i + mu * v_i)
                   for param_i, grad_i, v_i in zip(self.params, grads, self.velocities)]

        updates = [param_updates[i] for i in attack_layers]

        monmen_updates = [(v_i, mu * v_i - learning_rate * grad_i)
                    for grad_i, v_i in zip(grads, self.velocities)]

        updates += [monmen_updates[i] for i in attack_layers]

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



def find_ad_parameter(nn, target_image, target_class, attack_layers):
    '''
    find the most closed adversarial parameters by GD
    :param gradsfunc:
    :param proby: calculate the probability of target class given the input
    :param target_image: initial input
    :param target_class:
    :return:
    '''
    input_pattern = [numpy.copy(target_image)]
    proby = nn.proby
    deriv_y2param_func = nn.deriv_y2param_func

    epoch = 0
    while epoch < 1000:
        grads_by_parames = deriv_y2param_func(input_pattern, target_class)
        probyvalue, cur_y = proby(input_pattern, target_class)
        probyvalue = numpy.exp(probyvalue)

        if epoch % 10 == 0:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))

        if cur_y == target_class:
            print("Epoch %i : confidence is %e" % (epoch, probyvalue))
            return True

        # updates on selected layers
        for layer in attack_layers:
            # we update every perturbation, later we will minimize the FI count
            nn.params[layer].set_value(numpy.asarray(nn.params[layer].get_value())+numpy.asarray(grads_by_parames[layer])*0.01)
        epoch += 1
    return False


def eval_adversarial_paramters(outputfile,attack_layers):
    '''
    Evaluate the adversarial parameter for MNIST set
    :param fname:
    :param gradsfunc:
    :param proby:
    :param folder:
    :return:
    '''

    # Initialize the neural network
    nn = LeNet(batch_size=1, dropout_rate=-1)
    f = open('./Compression/NO_REGULAR/connection_count_431080.0.pkl', 'rb')
    load_value = pickle.load(f)
    nn.resume_all(load_value[0], load_value[1])
    nn.log_init()
    f.close()
    proby = nn.proby

    # Load the test set
    dataf = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(dataf)
    dataf.close()
    train_x = train_set[0]
    train_y = train_set[1]

    count = 0
    for i in range(10):
        nn.init_by_log()
        # whether current input pattern is correctly predicted
        ori_img = numpy.asarray(train_x[i], dtype=theano.config.floatX)

        _, ori_pred_class = proby([ori_img], 0)
        if ori_pred_class != train_y[i]:
            continue
        # create adversarial predict targets
        target_class_list = range(10)
        target_class_list.remove(train_y[i])
        for target_class in target_class_list:
            if find_ad_parameter(nn, ori_img, target_class, attack_layers):
                result=[i, ori_img, ori_pred_class, target_class, nn.params, nn.get_accuracy()]
                f = open(outputfile[0]+str(count)+outputfile[1], 'wb')
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
                count += 1




if __name__ == '__main__':

    theano.sandbox.cuda.use("gpu"+sys.argv[1])

    if not os.path.exists('./Fault_Injection/'):
        os.mkdir('./Fault_Injection/')


    # Load the train set
    dataf = gzip.open('mnist.pkl.gz', 'rb')
    train_set, _, _ = pickle.load(dataf)
    dataf.close()
    train_x = train_set[0]
    train_y = train_set[1]


    f = open('./Compression/NO_REGULAR/connection_count_431080.0.pkl', 'rb')
    load_value = pickle.load(f)
    f.close()
    normal_param = load_value[0]

    #Single Pattern Attack By SGD
    # nn = LeNet(cf_type='no_regular')
    # for attack_layers, fname in [[[0, 1], '0_1'],[[2, 3], '2_3'],[[4, 5], '4_5'],[[6, 7], '6_7'], [range(8), 'all']]:
    #     fig, (ax0, ax1) = plt.subplots(ncols=2)
    #     # eval_adversarial_paramters(['./Fault_Injection/Single_Pattern_Attack/','_attack_layers_'+fname+'.pkl'], attack_layers)
    #     FI_num = []
    #     FI_overallaccuracy = []
    #     for i in range(90):
    #         print(fname+'_'+str(i))
    #         f = open('./Fault_Injection/Single_Pattern_Attack/'+str(i)+'_attack_layers_'+fname+'.pkl')
    #         ad_result = pickle.load(f)
    #         f.close()
    #         cur_FI_num, cur_overaccu = find_fault_injection_points(nn, normal_param, ad_result[4], ad_result[3], train_x[ad_result[0]])
    #         FI_num.append(cur_FI_num)
    #         FI_overallaccuracy.append(cur_overaccu)
    #
    #     ax0.hist(FI_num)
    #     ax1.hist(FI_overallaccuracy)
    #     fig.savefig('./Fault_Injection/Single_Pattern_Attack/attack_layers_'+fname+'.pdf')

    # Single Pattern Attack by train
    for attack_sample in range(10):
        target_class_list = range(10)
        target_class_list.remove(train_y[attack_sample])

        for target_class in target_class_list:
            for attack_layers, fname in [[[0, 1], '0_1'],[[2, 3], '2_3'],[[4, 5], '4_5'],[[6, 7], '6_7'], [range(8), 'all']]:
                dataset = load_adversarial_augment_data('mnist.pkl.gz', train_x[attack_sample], target_class)
                nn = LeNet_Adversarial_Train(dataset=dataset,cf_type='minimize_FI',attack_layers=attack_layers)
                f = open('./Compression/NO_REGULAR/connection_count_431080.0.pkl', 'rb')
                load_value = pickle.load(f)
                f.close()
                nn.resume_all(load_value[0], load_value[1])
                nn.log_init()
                nn.train(40)

                # whether current input pattern is correctly predicted
                ori_img = numpy.asarray(train_x[attack_sample], dtype=theano.config.floatX)
                ori_img = numpy.tile(ori_img, (500, 1))
                _, cur_predict = nn.proby(ori_img, 0)

                if cur_predict[0] == target_class:
                    result = [attack_sample, ori_img, train_y[attack_sample], target_class, nn.params, nn.get_accuracy()]
                    f = open('./Fault_Injection/Single_Pattern_Attack_byTrain_optFINUM/' + str(attack_sample) + '_attack_layers_'+fname+'.pkl', 'wb')
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()




    # Single Class Attack by train
    # for attack_sample in range(5):
    #     ad_dict,_ = generate_adver_dict(10,1)
    #     for attack_layers, fname in [[[0, 1], '0_1'],[[2, 3], '2_3'],[[4, 5], '4_5'],[[6, 7], '6_7'], [range(8), 'all']]:
    #         #TODO attack layers realated modification
    #         dataset = load_adversarial_data('mnist.pkl.gz', ad_dict)
    #         nn = LeNet_Adversarial_Train(dataset=dataset,cf_type='no_regular',attack_layers=attack_layers)
    #         f = open('./Compression/NO_REGULAR/connection_count_431080.0.pkl', 'rb')
    #         load_value = pickle.load(f)
    #         f.close()
    #         nn.resume_all(load_value[0], load_value[1])
    #         nn.log_init()
    #         nn.train(40)
    #
    #         #TODO class wise accuracy
    #         result = [ad_dict,nn.params,nn.get_prediction_detail()]
    #         f = open('./Fault_Injection/Single_Class_Attack/' + str(attack_sample) + '_attack_layers_'+fname+'.pkl', 'wb')
    #         pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    #         f.close()

