from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import random
import theano.sandbox.cuda
from LeNet_resume import LeNet
from LeNet_resume import obtain_grayscale
import matplotlib.cm as cm


class LeInspect(LeNet):
    def getLayerOutFunc(self):
        layersout = theano.function([self.x], [self.layer0.output, self.layer1.output, self.layer2.output],allow_input_downcast=True)
        return layersout

    def getgrad_O_wrt_layer(self, cindex):
        # layer0grad = T.grad(self.layer3.p_y_given_x[0][cindex], self.layer0.output)
        # layer1grad = T.grad(self.layer3.p_y_given_x[0][cindex], self.layer1.output)
        layer2grad = T.grad(self.layer3.p_y_given_x[0][cindex], self.layer2.output)
        # layergrad = theano.function([self.x], [layer0grad,layer1grad,layer2grad])
        layergrad = theano.function([self.x], layer2grad)
        return layergrad

def obtain_image_set(data_file):
    dataf = open(data_file, 'rb')
    result = pickle.load(dataf)

    target_dict = {}
    for i in result:
        if target_dict.has_key(i[0][1]):
            target_dict[i[0][1]].append(i[1])
        else:
            target_dict[i[0][1]] = [i[1]]
    return target_dict


def normalize(seq):
    max_e = numpy.max(seq)
    min_e = numpy.min(seq)
    seq = (seq - min_e)/(max_e - min_e)
    return  seq


def inspect_internal_neurons():
    theano.sandbox.cuda.use("gpu0")
    model = 'RandInit'
    # Reload the LeNet parameters
    f = open('./weight/normal_weights_' + model + '.pkl', 'rb')
    normal_params = pickle.load(f)
    f.close()

    # Resume LeNet
    lenn = LeInspect(normal_params)
    layerout = lenn.getLayerOutFunc()
    colors = cm.rainbow(numpy.linspace(0, 1, 20))

    neuron_activation = [[] for _ in range(12)]
    neuron_grad = [[] for _ in range(12)]
    image_set = obtain_image_set('./eval_efforts/Constraint_mnist_GDBack_' + model + '.pkl')
    for digit in [8]:
        layergrads = lenn.getgrad_O_wrt_layer(digit)
        for ind in range(len(image_set[digit])):
            i = image_set[digit][ind]
            inlayer = layerout([i[1][0]])
            layervalue_t = inlayer[0].flatten().tolist() + inlayer[1].flatten().tolist() + inlayer[2].flatten().tolist()
            inlayer = layerout([i[0]])
            layervalue_o = inlayer[0].flatten().tolist() + inlayer[1].flatten().tolist() + inlayer[2].flatten().tolist()
            internal_layer = [abs(layervalue_o[ii] - layervalue_t[ii]) for ii in range(len(layervalue_o))]

            inlayer = layergrads([i[0]])
            layergradvalue = inlayer[0].flatten().tolist() + inlayer[1].flatten().tolist() + inlayer[
                2].flatten().tolist()

            # plt.plot(internal_layer,'.')
            # plt.savefig('./neuron_activation/Activation_Change_RandomLastInit/cifar'+str(ind)+'_8'+'.png')
            # plt.close()
            neuron_activation[digit].append(internal_layer)
            neuron_grad[digit].append(layergradvalue)

        intern_neuron = numpy.asarray(neuron_activation[digit])
        intern_neuron = numpy.mean(intern_neuron, axis=0)
        intern_grads = numpy.asarray(neuron_grad[digit])
        intern_grads = numpy.mean(intern_grads, axis=0)
        # intern_neuron[0:2880] = normalize(intern_neuron[0:2880])
        # intern_neuron[2880:3680] = normalize(intern_neuron[2880:3680])
        # intern_neuron[3680:] = normalize(intern_neuron[3680:])
        intern_neuron = intern_neuron.tolist()
        intern_grads = intern_grads.tolist()
        intern_neuron = zip(intern_neuron, intern_grads)
        intern_neuron[0:2880] = sorted(intern_neuron[0:2880], key=lambda x: x[0])
        intern_neuron[2880:3680] = sorted(intern_neuron[2880:3680], key=lambda x: x[0])
        intern_neuron[3680:] = sorted(intern_neuron[3680:], key=lambda x: x[0])
        intern_grads = map(lambda x: abs(x[1]), intern_neuron)
        intern_neuron = map(lambda x: x[0], intern_neuron)

        intern_neuron[0:2880] = normalize(intern_neuron[0:2880])
        intern_neuron[2880:3680] = normalize(intern_neuron[2880:3680])
        intern_neuron[3680:] = normalize(intern_neuron[3680:])

        intern_grads[0:2880] = normalize(intern_grads[0:2880])
        intern_grads[2880:3680] = normalize(intern_grads[2880:3680])
        intern_grads[3680:] = normalize(intern_grads[3680:])

        # for fil in range(20):
        #     intern_neuron[fil*144:(fil+1)*144] = sorted(intern_neuron[fil*144:(fil+1)*144])
        # for fil in range(50):
        #     intern_neuron[2880+fil*16:2880+(fil+1)*16] = sorted(intern_neuron[2880+fil*16:2880+(fil+1)*16])
        plt.plot(intern_neuron, color=colors[digit * 2], label=str(digit) + '_activation', marker='.', linestyle='')
        plt.plot(intern_grads, color=colors[digit * 2 + 1], label=str(digit) + '_grad', marker='.', linestyle='')

    plt.axvline(2880 - 1, color='g', linestyle='--')
    for i in range(1, 51):
        plt.axvline(2880 + i * 16 - 1, color='g', linestyle='--')
    plt.legend()

    plt.show()

def inspect_neuron_input_quality(params,model,colors):
    lenn = LeInspect(params)
    layerout = lenn.getLayerOutFunc()

    dataf = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(dataf)
    dataf.close()
    test_x = test_set[0]
    test_y = test_set[1]
    # Count the neuron activation values for test set
    neuron_values = [[] for _ in range(10)]
    for ind in range(200):
        ori_img = numpy.asarray(test_x[ind], dtype=theano.config.floatX)
        inlayer = layerout([ori_img])
        # layervalue = numpy.reshape(inlayer[0], (20, 144))
        # layervalue = numpy.mean(layervalue, axis=1)
        layervalue = inlayer[2].flatten().tolist()
        neuron_values[test_y[ind]].append(layervalue)
    neuron_values_mean = numpy.zeros((10, len(layervalue)))
    neuron_values_std = numpy.zeros((10, len(layervalue)))
    for tc in range(10):
        neuron_values[tc] = numpy.asarray(neuron_values[tc])
        neuron_values_mean[tc, :] = numpy.mean(neuron_values[tc], axis=0)
        neuron_values_std[tc, :] = numpy.std(neuron_values[tc], axis=0)
    neuron_mean_std = numpy.std(neuron_values_mean, axis=0)/numpy.max(neuron_values_std,axis=0)
    plt.hist(neuron_mean_std.tolist(),bins=25,label=model,histtype='step',color=colors)

def inspect_neuron_output_quality(params, model, colors):
        lenn = LeInspect(params)

        dataf = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(dataf)
        dataf.close()
        test_x = test_set[0]
        test_y = test_set[1]
        # Count the grads w.r.t each output neuron
        neuron_grads = [[] for _ in range(10)]
        for digit in range(10):
            for ind in range(200):
                layergrads = lenn.getgrad_O_wrt_layer(digit)
                ori_img = numpy.asarray(test_x[ind], dtype=theano.config.floatX)
                inlayer_grad = layergrads([ori_img])
                layervalue = inlayer_grad.flatten().tolist()
                neuron_grads[digit].append(layervalue)
        neuron_values_mean = numpy.zeros((10, len(layervalue)))
        neuron_values_std = numpy.zeros((10, len(layervalue)))
        for tc in range(10):
            neuron_grads[tc] = numpy.asarray(neuron_grads[tc])
            neuron_values_mean[tc, :] = numpy.mean(neuron_grads[tc], axis=0)
            neuron_values_std[tc, :] = numpy.std(neuron_grads[tc], axis=0)
        # neuron_mean_std = numpy.std(neuron_values_mean, axis=0) / numpy.max(neuron_values_std, axis=0)
        # neuron_mean_std = numpy.std(neuron_values_mean, axis=0)
        return neuron_grads
        plt.hist(numpy.absolute(neuron_values_mean.flatten()).tolist(), bins=25, label=model, histtype='step', color=colors)


if __name__ == '__main__':

    theano.sandbox.cuda.use("gpu0")
    models = [
    "RandInit_Contract5_all_e0",
    "RandInit_Contract5_e0_L2_en2_all",
    "RandInit_FGS_0.1",
    "RandInit_L2R_en2",
    "RandInit"]

    colors = cm.rainbow(numpy.linspace(0, 1, 5))

    count = 0
    grad_dict = {}
    for model in models:
    # Reload the LeNet parameters
        f = open('./weight/normal_weights_'+model+'.pkl', 'rb')
        normal_params = pickle.load(f)
        f.close()
        neurons_grads = inspect_neuron_output_quality(normal_params,model,colors[count])
        grad_dict[model] = neurons_grads

        #Plot weight

        # for layerind in [0]:
        #     layerweight = numpy.absolute(normal_params[layerind*2].get_value().flatten())
        #     layerweight /= numpy.max(layerweight)
        #     layerbias = normal_params[layerind*2+1].get_value().flatten()
        #     plt.hist(layerweight, bins=20,label=model,histtype='step',normed=True,color=colors[count])
        count += 1
        # plt.legend()
        # plt.savefig('./neuron_activation/Weight/'+model+'.pdf')
        # plt.close()
    #Count output weight in next layers
    # output_weight = normal_params[4].get_value()
    # output_bias = normal_params[1].get_value()
    f = open('./neuron_activation/L2_neuron_grads.pkl','wb')
    pickle.dump(grad_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    # colors = cm.rainbow(numpy.linspace(0, 1, 10))
    # for tc in [1,8]:
        # rest_class = range(10)
        # rest_class.remove(tc)
        # rest_record = output_weight[:, rest_class]
        # rest_record_max = numpy.amax(rest_record,axis=1).tolist()
        # rest_record_min = numpy.amin(rest_record,axis=1).tolist()
        # rest_record_mean = numpy.mean(rest_record, axis=1).tolist()
        # list_record = output_weight[:, tc].tolist()
        # three_list = zip(rest_record_min,list_record,rest_record_max,rest_record_mean)
        # diff_list = map(lambda x: abs(x[1]-x[3]),three_list)
        # diff_list = sorted(diff_list)
        # three_list = sorted(three_list,key=lambda x:x[1])
        # rest_record_min = map(lambda x:x[0],three_list)
        # rest_record_max = map(lambda x: x[2], three_list)
        # rest_record_mean = map(lambda x: x[3], three_list)
        # list_record = map(lambda x: x[1], three_list)
        # list_record = sorted(list_record)
        # plt.plot(list_record,color=colors[0],marker='.', linestyle='', label='class {i}'.format(i=tc))
        # plt.plot(rest_record_max, color=colors[5],marker='.', label='max',linestyle='')
        # plt.plot(rest_record_min, color=colors[9],marker='.', label='min',linestyle='')
        # plt.plot(rest_record_mean, color=colors[9],marker='.', label='mean',linestyle='')
        # plt.plot(diff_list, color=colors[tc], marker='.', label='diff {i}'.format(i=tc), linestyle='')
        # plt.legend()
    # plt.legend()
    # plt.show()

