import matplotlib.pyplot as plt
import cPickle
import numpy
import os
import theano
import numpy
import seaborn as sns


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm


#######
# Draw Weight Distribution
#######
def draw_weight_distribution(folder):
    files = os.listdir(folder)
    files = filter(lambda a: 'connection' in a, files)
    for i in range(len(files)):
        files[i] = int(files[i].split("_")[2].split('.')[0])
    files = sorted(files,reverse=True)
    files = filter(lambda a: a > 1000,files)
    colors = cm.rainbow(numpy.linspace(0, 1, len(files)+1))

    for i in [0,2,4,6]:
        plt.figure()
        count = 0
        for fname in files:
            f = open(folder+'/connection_count_' + str(fname)+'.0.pkl', 'rb')
            load_value = cPickle.load(f)
            f.close()

            # for i in range(8):
            if True:
                weight = load_value[0][i].get_value()
                mask = load_value[1][i].get_value()
                treal_weight = weight * mask
                real_weight = list(numpy.reshape(treal_weight,(-1)))
                real_weight = filter(lambda a: a != 0, real_weight)
                print('%d,%d'%(fname,len(real_weight)))
            sns.distplot(real_weight, kde=True, hist=False, label=str(fname), kde_kws={"color": colors[count]})
            count += 1
        plt.legend(loc='top left')
        plt.savefig(folder+'/layer_'+str(3-i/2)+'.png')


########
# Get Accuracy VS NN Size Under Fault
########
def get_accuracy_vs_NNsize(net,folder=''):
    f = open("./"+net+"/Compression_Result/reliability_"+folder+'_'+str(0)+'.pkl','rb')
    result = cPickle.load(f)
    f.close()
    x = []
    y = []
    for k, v in result.items():
        x.append(k)
        y.append(1 - numpy.mean(v))
    xy = zip(x, y)
    xysort = sorted(xy, key=lambda it: it[0], reverse=True)
    x = map(lambda it: it[0], xysort)
    y = map(lambda it: it[1], xysort)

    return [x,y]


########
# Draw Accuracy VS NN Size Under Fault
########
def draw_accuracy_vs_NNsize():
    f = open("./Compression_Result/reliability"+str(0)+'.pkl','rb')
    result = cPickle.load(f)
    f.close()
    x = []
    y = []
    for k, v in result.items():
        x.append(k)
        y.append(1 - numpy.mean(v))
    xy = zip(x, y)
    xysort = sorted(xy, key=lambda it: it[0], reverse=True)
    x = map(lambda it: it[0], xysort)
    ynormal = map(lambda it: it[1], xysort)


    # for i in [0,0.1,0.01,0.001,0.0001,0.00001]:
    for i in [0]:
        f = open("./Compression_Result/reliability"+str(i)+'.pkl','rb')
        result = cPickle.load(f)
        f.close()
        x = []
        y = []
        for k,v in result.items():
            x.append(k)
            y.append(1-numpy.mean(v))
        xy = zip(x,y)
        xysort = sorted(xy,key=lambda it: it[0],reverse=True)
        x = map(lambda it: it[0],xysort)
        y = map(lambda it: it[1],xysort)
        # ydegrad = []
        # for ind in range(len(ynormal)):
        #     ydegrad.append((ynormal[ind]-y[ind])/ynormal[ind])
        plt.plot(x,y,label=str(i))
    plt.gca().invert_xaxis()
    plt.legend(loc=3)
    plt.axvline(13080,linestyle='--')
    # plt.xlim(left=20000)
    # plt.show()

def draw_wegith_pruning_for_each_layer(folder,layer):
    '''
    This function plots how weights and biases are pruned for each layer during compression.
    :return:
    '''
    files = os.listdir(folder)
    files = filter(lambda x: 'connection_count' in x, files)
    for i in range(len(files)):
        files[i] = int(files[i].split("_")[2].split('.')[0])
    files = sorted(files)
    colors = cm.rainbow(numpy.linspace(0, 1, 8))
    maskratio = [[] for _ in range(8)]
    for fname in files:
        f = open(folder+'/connection_count_' + str(fname)+'.0.pkl', 'rb')
        load_value = cPickle.load(f)
        f.close()

        for i in [layer]:
            layer_mask = load_value[1][i].get_value()
            layer_mask = numpy.reshape(layer_mask,(-1))
            effective_item = layer_mask.sum()
            total_item = layer_mask.shape[0]
            ratio = (effective_item*1.)/total_item
            maskratio[i].append(ratio)
    for i in [layer]:
        plt.plot(files,maskratio[i],color=colors[i],label='layer_'+str(3-i/2)+'_'+str(i%2))
        # plt.text(files[i*3+10],maskratio[i][i*3+10],'layer_'+str(3-i/2)+'_'+str(i%2))
    # plt.gca().invert_xaxis()
    # plt.legend(loc='upper left')
    # plt.show()

def draw_weight_grads_distribution():
    '''
    draw_weight_gradstopp_distribution()
    :return:
    '''
    files = os.listdir('./Compression/L2EN3/Selected')
    files = filter(lambda x: 'connection_count' in x,files)
    for i in range(len(files)):
        files[i] = int(files[i].split("_")[2].split('.')[0])
    files = sorted(files)


    for fname in files:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        f = open('./Compression/L2EN3/Selected/connection_count_' + str(fname)+'.0.pkl', 'rb')
        load_value = cPickle.load(f)
        f.close()
        normal_params = load_value[0]
        gradtoPP = load_value[3]

        flatten_params = []
        flatten_grads  = []
        for i in [0,2,4,6]:
            params_group = numpy.reshape(normal_params[i].get_value(), (-1,))
            grads_group = numpy.reshape(numpy.asarray(gradtoPP[i]),(-1,))
            flatten_params += list(params_group)
            flatten_grads += list(grads_group)
        # flatten_params = map(lambda x: abs(x),flatten_params)
        # flatten_grads = map(lambda x: abs(x), flatten_grads)
        xleft = (min(flatten_params)-max(flatten_params))*0.1
        ybot = (min(flatten_grads)-max(flatten_grads))*0.1
        ax.plot(flatten_params,flatten_grads,linestyle='None',marker='o')
        ax.set_xlabel('params')
        ax.set_ylabel('grads')
        # ax.set_xlim(left=xleft)
        # ax.set_ylim(bottom=ybot)

        fig.savefig('./Compression/L2EN3/Selected/' + str(fname)+'.png')

def LeNet_Time_Channel(config):
    '''
    Draw the time elapse for each class in LeNet
    :param folder:
    :return:
    '''
    f = open('./LeNet/Time_Channel/' + config + '_.pkl', 'rb')
    time_result = cPickle.load(f)
    f.close()
    colors = cm.rainbow(numpy.linspace(0, 1, 11))
    for k,v in time_result.items():
        if k != 351080:
            continue
        time_elapse = [[] for _ in range(10)]
        for sample in v:
            if sample[1][0] == sample[0]:
                time_elapse[sample[1][0]].append(sample[2])
        plt.figure()
        for i in range(len(time_elapse)):
            if len(time_elapse[i])!= 0:
                sns.distplot(time_elapse[i], kde=True, hist=False, label=str(i), kde_kws={"color": colors[i]})
        plt.show()
        # plt.savefig('./LeNet/Time_Channel/'+config+'_'+str(k)+'.png')
        # plt.close()

if __name__=='__main__':
    # draw_wegith_pruning_for_each_layer()
    # draw_accuracy_vs_NNsize()
    # draw_weight_distribution('./LeNet/Compression/LOGINIT_NO_REGULAR/Selected')
    # draw_wegith_pruning_for_each_layer('./LeNet/Compression/NO_REGULAR/Selected')
    layer = 0
    for folder in [
        'NO_REGULAR',
        # 'LOGINIT_NO_REGULAR_0.05',
        'LOGINIT_NO_REGULAR_0.1',
        #  'LOGINIT_NO_REGULAR_0.15'
        # 'CONTRACT_LIKE',
        # 'CONTRACT_LIKE_L2EN3',
        'L2EN3',
        'LOGINIT_L2EN3',
        # 'CC_NO_REGULAR',
        # 'CC_LOGINIT_NO_REGULAR',
        # 'LOGINIT_CONTRACT_LIKE',
        # 'FC_LOGINIT_CONTRACT_LIKE',
        # 'CC_CONTRACT_LIKE'
        # 'LBL_LOGINIT_NO_REGULAR',
        # 'PCC_LOGINIT_NO_REGULAR',
        # 'FC_LOGINIT_NO_REGULAR_40',
        # 'LOGINIT_NO_REGULAR',
        # 'MIX_LOGINIT_L2_NO_REGULAR',
        # 'MIX_LOGINIT_NO_REGULAR_L2'
                   ]:
        draw_wegith_pruning_for_each_layer('./LeNet/Compression/'+folder+'/Selected',layer)

    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.savefig(str(layer)+'.pdf')