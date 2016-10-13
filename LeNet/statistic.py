import matplotlib.pyplot as plt
import cPickle
import matplotlib
import numpy
import os
import theano
import numpy
# import seaborn as sns
import matplotlib.cm as cm


#######
# Draw Weight Distribution
#######
def draw_weight_distribution():
    files = os.listdir('./Compression/Selected/')
    for i in range(len(files)):
        files[i] = int(files[i].split("_")[2].split('.')[0])
    files = sorted(files)
    colors = cm.rainbow(numpy.linspace(0, 1, len(files)+1))

    count = 0
    for fname in files:
        f = open('./Compression/Selected/connection_count_' + str(fname)+'.0.pkl', 'rb')
        load_value = cPickle.load(f)
        f.close()
        real_weight = []
        for i in range(8):
            weight = load_value[0][i].get_value()
            mask = load_value[1][i].get_value()
            treal_weight = weight * mask
            real_weight += list(numpy.reshape(treal_weight,(-1)))
        real_weight = filter(lambda a: a!=0, real_weight)
        real_weight = map(lambda a:abs(a),real_weight)

        sns.distplot(real_weight,kde=False,label=str(fname),kde_kws={"color": colors[count]},hist_kws={"color":colors[count]},norm_hist=False)
        count += 1
    plt.legend(loc='top left')
    plt.show()


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

def draw_wegith_pruning_for_each_layer():
    '''
    This function plots how weights and biases are pruned for each layer during compression.
    :return:
    '''
    files = os.listdir('./Compression/L2EN3/Selected/')
    files = filter(lambda x: 'connection_count' in x, files)
    for i in range(len(files)):
        files[i] = int(files[i].split("_")[2].split('.')[0])
    files = sorted(files)
    colors = cm.rainbow(numpy.linspace(0, 1, 8))
    maskratio = [[] for _ in range(8)]
    for fname in files:
        f = open('./Compression/L2EN3/Selected/connection_count_' + str(fname)+'.0.pkl', 'rb')
        load_value = cPickle.load(f)
        f.close()

        for i in range(8):
            layer_mask = load_value[1][i].get_value()
            layer_mask = numpy.reshape(layer_mask,(-1))
            masked_item = layer_mask.sum()
            total_item = layer_mask.shape[0]
            ratio = 1 - masked_item*1./total_item
            maskratio[i].append(ratio)
    for i in range(8):
        plt.plot(files,maskratio[i],color=colors[i],label='layer_'+str(3-i/2)+'_'+str(i%2))
        plt.text(files[i*3+10],maskratio[i][i*3+10],'layer_'+str(3-i/2)+'_'+str(i%2))
    plt.gca().invert_xaxis()
    plt.legend(loc='upper left')
    plt.show()

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

if __name__=='__main__':
    # draw_wegith_pruning_for_each_layer()
    draw_accuracy_vs_NNsize()

