import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def plot_wb_distribution(paramfile,  imagefile, derivativefile=0):

    f = open('0_0_weights.pkl', 'rb')
    normal_params = pickle.load(f)
    f.close()

    f = open(paramfile, 'rb')
    adver_params = pickle.load(f)
    f.close()

    f = open(derivativefile, 'rb')
    adver_derivative = pickle.load(f)
    f.close()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 10))

    for i in range(4):

        worb = np.reshape((normal_params[2*i].get_value()-adver_params[2*i].get_value()), (-1))
        k = np.reshape(adver_derivative[2*i], (-1))
        worb = worb*k
        axes[0, i].hist(worb, 40)
        axes[0, i].set_title('Weight_Layer'+str(3-i), fontsize=10)
        # axes[0, i].set_xticks([m*1.0/100 for m in range(-8, 9)])

        worb = np.reshape(normal_params[2*i+1].get_value()-adver_params[2*i+1].get_value(), (-1))
        k = np.reshape(adver_derivative[2*i+1], (-1))
        worb = worb*k
        axes[1, i].hist(worb, 40)
        axes[1, i].set_title('Bias_Layer'+str(3-i), fontsize=10)
        # axes[1, i].set_xticks([m * 1.0 / 100 for m in range(-4, 5)])

    plt.savefig(imagefile)
    plt.close()


def plot_wb_diff():
    f = open('0_0_weights.pkl', 'rb')
    normal_parameter = pickle.load(f)
    f.close()

    flatten_normp = []
    for i in range(8):
        normal_group = np.reshape(normal_parameter[i].get_value(), (-1))
        flatten_normp += list(normal_group)

    colors = cm.rainbow(np.linspace(0,1,100))
    for i in range(210,290):
        f = open('../multi_miss/adver_map_set'+str(i)+'_weights.pkl', 'rb')
        adver_parameter = pickle.load(f)
        f.close()
        flatten_adverp = []
        for j in range(8):
            adver_group = np.reshape(adver_parameter[j].get_value(), (-1))
            flatten_adverp += list(adver_group)

        flatten_diff = [abs(flatten_normp[k] - flatten_adverp[k]) for k in range(len(flatten_normp))]
        flatten_diff.sort()
        plt.plot(flatten_diff, color=colors[i/10+1], label=str(i/10+1))
    plt.show()


def draw_accuracy_degrade_by_diff():
    f = open('ac_degrade_by_diff_smallfirst_adver_dict.pkl','rb')
    adver_dict_set = pickle.load(f)
    f.close()
    for adver_map in adver_dict_set:
        adver_map[0].sort()
        i = adver_map[0][0]
        j = adver_map[0][1]
        prefix = str(i) + '_' + str(j) + '_weights'
        dprefix = str(i) + '_' + str(j) + '_deriative'
        # f = open(str(i) + '_' + str(j) + '_ac_degrade_by_diff_gradient_abs.pkl', 'rb')
        f = open(str(i) + '_' + str(j) + '_ac_degrade_by_diff_smallfirst.pkl', 'rb')
        th_vs_accuracy = pickle.load(f)
        f.close()

        # f = open(str(i) + '_' + str(j) + '_ac_degrade_by_diff_gradient2.pkl', 'rb')
        # f = open(str(i) + '_' + str(j) + '_ac_degrade_by_diff_gradient_abs2.pkl', 'rb')
        # th_vs_accuracy += pickle.load(f)
        # f.close()

        # x = [431080 - e[0] for e in th_vs_accuracy]
        x = [e[0] for e in th_vs_accuracy]
        y = [e[2] for e in th_vs_accuracy]
        tcomxy = [[x[k],y[k]] for k in range(len(x))]
        tcomxy.sort(key=lambda k: k[0])
        comxy = []
        for i in tcomxy:
            if i[0] != -1:
                comxy.append(i)

        x = [comxy[k][0] for k in range(len(comxy))]
        y = [comxy[k][1] for k in range(len(comxy))]
        drawx = []
        drawy = []
        for k in range(len(x)):
            # if x[k] < 50000:
            drawx.append(x[k])
            drawy.append(y[k])
        plt.plot(drawx, drawy)
        # plot_wb_distribution(prefix+'.pkl', dprefix+'.pkl', './fig/'+prefix+'.pdf')
    plt.show()


if __name__ == '__main__':
    # draw_accuracy_degrade_by_diff()
    plot_wb_diff()