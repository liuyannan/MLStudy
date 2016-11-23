import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from statistic import get_accuracy_vs_NNsize
#Load the datasets
# f = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()
#
# train_x = numpy.asarray(train_set[0])
# train_y = numpy.asarray(train_set[1])
#
# first_image = numpy.reshape(train_x[0], (28, 28))
# print train_y
# plt.imshow(first_image,cmap='Greys_r')
# plt.show()

import six.moves.cPickle as pickle


def draw_scratch():
    data_dir = './scratch_zero/'

    pkllist = os.listdir(data_dir)

    for pkl in pkllist:
        f = open(data_dir+pkl, 'rb')
        result = pickle.load(f)
        f.close()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        # Transformed Image

        transform_img = [result[0][0]]
        image = result[2][0]
        probyvalue = result[0][1]
        probyvalue_diff = result[1][1]
        probyvalue_ori = result[2][1]

        gray_image = numpy.reshape(transform_img[0], (28, 28))
        axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1.0)
        axes[0].set_title('Score for class ' + ':' + str(probyvalue))

        # Difference
        gray_image = numpy.reshape(transform_img[0] - image, (28, 28))
        axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1.0)
        axes[1].set_title('Difference: score is ' + str(probyvalue_diff))

        # Original Image
        gray_image = numpy.reshape(image, (28, 28))
        axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=0, vmax=1.0)
        axes[2].set_title('Original: score ' + str(probyvalue_ori))

        plt.savefig(data_dir+pkl+'.png')
        plt.close()


def draw_ambi():
    data_dir = './ambi/'

    pkllist = os.listdir(data_dir)

    for pkl in pkllist:
        f = open(data_dir+pkl, 'rb')
        result = pickle.load(f)
        f.close()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        # Transformed Image

        transform_img = [result[0][0]]
        image = result[2][0]
        probyvalue1 = result[0][1]
        probyvalue2 = result[0][2]
        probyvalue1_ori = result[2][1]
        probyvalue2_ori = result[2][2]
        probyvalue1_diff = result[1][1]
        probyvalue2_diff = result[1][2]

        gray_image = numpy.reshape(transform_img[0], (28, 28))
        axes[0].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=image.min(), vmax=image.max())
        axes[0].set_title('class1 ' + ':' + str(probyvalue1)+'\nclass2 ' + ':' + str(probyvalue2))

        # Difference
        gray_image = numpy.reshape(transform_img[0] - image, (28, 28))
        axes[1].imshow(gray_image, cmap=plt.get_cmap('gray'))
        axes[1].set_title('class1 ' + ':' + str(probyvalue1_diff) + '\nclass2 ' + ':' + str(probyvalue2_diff))

        # Original Image
        gray_image = numpy.reshape(image, (28, 28))
        axes[2].imshow(gray_image, cmap=plt.get_cmap('gray'), vmin=image.min(), vmax=image.max())
        axes[2].set_title('class1 ' + ':' + str(probyvalue1_ori) + '\nclass2 ' + ':' + str(probyvalue2_ori))

        plt.savefig(data_dir+pkl+'.png')
        plt.close()

def obtain_grayscale(rgbimg):
    grayimage = numpy.empty([1024])
    for i in range(1024):
        grayimage[i] = (rgbimg[i] * 0.2989 + 0.5870 * rgbimg[i+1024] + 0.1140 * rgbimg[i+2048])/255.0
    grayimage.shape = (32, 32)
    grayimage = grayimage[2:30, 2:30]
    return grayimage

def plot_distortion_tc_2(data_file, evalbase='target'):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    target_dict = {}
    for i in result:
        if evalbase is 'target':
            basekey = i[0][1]
        else:
            basekey = i[0][0]
        # if i[0][0] != 3:
        #     continue
        if target_dict.has_key(basekey):
            target_dict[basekey].append(abs(i[1][0]-i[1][1][0]))
        else:
            target_dict[basekey] = [abs(i[1][0]-i[1][1][0])]
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    # for tc in range(10):
    for tc in [2]:
        if not target_dict.has_key(tc):
            continue
        tmp = numpy.asarray(target_dict[tc])
        tmp = numpy.mean(tmp,axis=0)
        tmp = tmp.reshape((-1,))
        tmp = tmp.tolist()
        tmp = sorted(tmp)
        plt.plot(tmp,color=colors[tc],label=str(tc))
        plt.text(len(tmp)-50,tmp[-50], evalbase+'_sample {i}'.format(i=tc)+str(data_file))
    # plt.legend()

def plot_distortion(data_file, evalbase='target'):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    target_dict = []
    for i in result:
        target_dict.append(abs(i[1][0]-i[1][1][0]))

    colors = cm.rainbow(numpy.linspace(0, 1, 10))

    tmp = numpy.asarray(target_dict)
    tmp = numpy.mean(tmp,axis=0)
    tmp = tmp.reshape((-1,))
    tmp = tmp.tolist()
    tmp = sorted(tmp)
    plt.plot(tmp,color=colors[0],label=str(0))
    plt.text(len(tmp)-50,tmp[-50], evalbase+'_sample {i}'.format(i=0)+str(data_file))
    # plt.legend()

def plot_every_distortion(data_file, evalbase='target'):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    target_dict = {}
    for i in result:
        if evalbase is 'target':
            basekey = i[0][1]
        else:
            basekey = i[0][0]
        # if i[0][0] != 3:
        #     continue
        if target_dict.has_key(basekey):
            target_dict[basekey].append(abs(i[1][0]-i[1][1][0]))
        else:
            target_dict[basekey] = [abs(i[1][0]-i[1][1][0])]
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    # for tc in range(10):
    for tc in [1]:
        if not target_dict.has_key(tc):
            continue
        distortion_array = numpy.asarray(target_dict[tc])
        for ind in range(10):
            tmp = distortion_array[ind]
            tmp = tmp.reshape((-1,))
            tmp = tmp.tolist()
            tmp = sorted(tmp)
            plt.plot(tmp,color=colors[tc],label=str(tc))
            plt.text(len(tmp)-50,tmp[-50], evalbase+'_sample {i}'.format(i=ind)+str(data_file))
    # plt.legend()

def get_distortion(data_file, evalbase='target'):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    target_dict = {}
    for i in result:
        if evalbase is 'target':
            basekey = i[0][1]
        else:
            basekey = i[0][0]
        # if i[0][0] != 3:
        #     continue
        if target_dict.has_key(basekey):
            target_dict[basekey].append(abs(i[1][0]-i[1][1][0]))
        else:
            target_dict[basekey] = [abs(i[1][0]-i[1][1][0])]
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    distortion_array = []
    for tc in range(10):
        tmp = numpy.asarray(target_dict[tc])
        distortion_array.append(tmp)
    return distortion_array

def distortion_L2(data_file):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    distortion = []
    effemask = []

    effcount = 0
    for i in result:
        tmp_norm = numpy.linalg.norm(i[1][0]-i[1][1][0])
        distortion.append(tmp_norm)
        if len(i[0]) == 3:
            if numpy.isnan(numpy.sum(i[1][1][0])) or i[0][2] != i[0][1]:
                effemask.append(0)
                distortion[-1] = -1
                continue
        effemask.append(1)
        effcount += 1
    effratio = (effcount * 1.)/len(result)

    tmp = numpy.asarray(distortion)
    mtmp = numpy.mean(tmp)

    return [mtmp,effratio,distortion,effemask]


def check_output_range(data_file, evalbase='target'):
    dataf = open(data_file,'rb')
    result = pickle.load(dataf)

    target_dict = {}
    for i in result:
        if evalbase is 'target':
            basekey = i[0][1]
        else:
            basekey = i[0][0]
        # if i[0][0] != 3:
        #     continue
        if target_dict.has_key(basekey):
            target_dict[basekey].append(abs(i[1][0]-i[1][1][0]))
        else:
            target_dict[basekey] = [abs(i[1][0]-i[1][1][0])]
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    for tc in range(10):
    # for tc in [8]:
        if not target_dict.has_key(tc):
            continue
        tmp = numpy.asarray(target_dict[tc])
        tmpmax = numpy.max(tmp,axis=0)
        tmpmin = numpy.min(tmp,axis=0)
        tmpmax = tmpmax.reshape((-1,))
        tmpmax = tmpmax.tolist()
        tmpmin = tmpmin.reshape((-1,))
        tmpmin = tmpmin.tolist()
        plt.plot(tmpmin,color=colors[tc],label=str(tc))
        plt.text(len(tmpmin)-50,tmpmin[-50], evalbase+'_sample {i}'.format(i=tc)+str(data_file))
        plt.plot(tmpmax, color=colors[tc], label=str(tc))
        plt.text(len(tmpmax) - 50, tmpmax[-50], evalbase + '_sample {i}'.format(i=tc) + str(data_file))
    # plt.legend()

def check_distortion_chage():
    RandInit_dist = get_distortion('./eval_efforts/Constraint_mnist_GDBack_RandInit.pkl',evalbase='source')
    L2R_dist = get_distortion('./eval_efforts/Constraint_mnist_GDBack_RandInit_L2R_en2.pkl',evalbase='source')
    colors = cm.rainbow(numpy.linspace(0, 1, 10))
    # for tc in range(10):
    for tc in [8]:
        for ind in range(RandInit_dist[tc].shape[0]):
            tmp = RandInit_dist[tc][ind]
            tmp = tmp.reshape((-1,))
            tmp = tmp.tolist()
            RandSorted = sorted(tmp)

            tmp = L2R_dist[tc][ind]
            tmp = tmp.reshape((-1,))
            tmp = tmp.tolist()
            L2RSorted = sorted(tmp)

            tmp = [L2RSorted[pix] - RandSorted[pix] for pix in range(len(L2RSorted))]
            plt.plot(tmp, color=colors[tc], label=str(tc))
            plt.text(len(tmp) - 50, tmp[-50], '_sample {i}'.format(i=tc))

def draw_distortion_direction(data_file):
    dataf = open(data_file, 'rb')
    result = pickle.load(dataf)

    direction_dict = {}
    for fc in range(10):
        for sc in range(10):
            direction_dict[(fc,sc)] = numpy.zeros((784,1))
    colors = cm.rainbow(numpy.linspace(0, 1, 10))

    for i in result:
        gradsvalue = i[1][1][0] - i[1][0]
        gradsvalue = gradsvalue.reshape(784,1)
        direction_dict[(i[0][0],i[0][1])] = numpy.concatenate((direction_dict[(i[0][0],i[0][1])],gradsvalue),axis=1)

    for k,v in direction_dict.items():

        tv = numpy.mean(v,axis=1)


        tv = numpy.reshape(tv,(28,28))
        plt.imshow(tv, cmap=plt.get_cmap('gray'), vmin=tv.min(), vmax=tv.max())
        plt.savefig('./pert_direction/' + str(k[0]) + '_' + str(k[1]) + '_img.pdf')
        # continue
        for i in range(784):
            rowdata = v[i,1:-1]
            plt.plot(numpy.ones(rowdata.shape)*i*10,rowdata,linestyle='',marker='.',ms=4)
        plt.ylim(-1.1, 1.1)
        plt.xlim(0,8000)
        plt.savefig('./pert_direction/'+str(k[0])+'_'+str(k[1])+'.pdf')
        plt.close()

def Cifar():
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    folders = [
        'NO_REGULAR',
        # 'CONTRACT_LIKE',
        # 'CONTRACT_LIKE_L2EN3',
        'L2',
        'LOGINIT_L2',
        # 'L2_RATIO'
        # 'CC_NO_REGULAR',
        # 'CC_LOGINIT_NO_REGULAR',
        # 'LOGINIT_CONTRACT_LIKE',
        # 'FC_LOGINIT_CONTRACT_LIKE',
        # 'CC_CONTRACT_LIKE'
        # 'PCC_LOGINIT_NO_REGULAR',
        # 'FC_LOGINIT_NO_REGULAR_40',
        'LOGINIT_NO_REGULAR'
    ]

    global_mask_exist = False
    global_distortion = []
    global_x =[]
    for folder in folders:
        files = os.listdir('./CifarNet/eval_efforts_rough/')
        files = filter(lambda a: "Constraint_mnist_GDBack_Compression_" + folder == '_'.join(a.split('_')[:-1]), files)
        models = map(lambda a: a.replace("Constraint_mnist_GDBack_Compression_" + folder, ""), files)

        # Directly draw the distortion result
        x = []
        y1 = []
        y2 = []

        # Only compare the same sample
        dist_matrix = []
        dist_mask = []

        for model in models:
            print(folder + model)
            if os.path.exists('./CifarNet/eval_efforts_rough/Constraint_mnist_GDBack_Compression_' + folder + model):
                x.append(float(model.split('_')[-1].split('.pkl')[0]))
                result = distortion_L2('./CifarNet/eval_efforts_rough/Constraint_mnist_GDBack_Compression_' + folder + model)
                y1.append(result[0])
                y2.append(result[1])
                dist_matrix.append(result[2])
                dist_mask.append(result[3])
            else:
                print(folder + model + 'not_found')

        global_distortion.append(dist_matrix)
        global_x.append(x)

        if not global_mask_exist:
            dist_mask = numpy.asarray(dist_mask)
            global_mask = numpy.prod(dist_mask,axis=0)
            global_mask_exist = True
        else:
            dist_mask = numpy.asarray(dist_mask)
            tmp_mask = numpy.prod(dist_mask,axis=0)
            global_mask *= tmp_mask

    # remove uncommon sample
    for i in range(len(folders)):
        folder = folders[i]
        dist_matrix = numpy.asarray(global_distortion[i])
        effective_samples = numpy.sum(global_mask)
        mask_expand = global_mask.reshape((1,global_mask.shape[0])).repeat(dist_matrix.shape[0],0)
        dist_matrix *= mask_expand
        dist_matrix = numpy.sum(dist_matrix,axis=1)/effective_samples
        y1 = list(numpy.squeeze(dist_matrix.reshape(1,-1)))
        x = global_x[i]

        x_sort, y1 = zip(*(sorted(zip(x, y1))))
        ax0.plot(x_sort, y1, label=folder)

        x, y = get_accuracy_vs_NNsize('CifarNet',folder)
        ax1.plot(x, y, label=folder)
    # ax1.set_ylim([0.9, 1])
    # plt.hlines(0,0,431080)
    plt.legend(loc=0)
    plt.gca().invert_xaxis()
    plt.show()
    exit()

def LeNet():

    fig, (ax0,ax1) = plt.subplots(nrows=2,sharex=True)
    for folder in [
                    'NO_REGULAR',
                   # 'LOGINIT_NO_REGULAR_0.05',
                   'LOGINIT_NO_REGULAR',
                   #  'LOGINIT_NO_REGULAR_0.15'
                    'CONTRACT_LIKE',
                    # 'CONTRACT_LIKE_L2EN3',
                    'L2EN3',
                    'LOGINIT_L2EN3',
                    # 'CC_NO_REGULAR',
                    # 'CC_LOGINIT_NO_REGULAR',
                    'LOGINIT_CONTRACT_LIKE',
                    # 'FC_LOGINIT_CONTRACT_LIKE',
                    # 'CC_CONTRACT_LIKE'
                    # 'LBL_LOGINIT_NO_REGULAR',
                    # 'PCC_LOGINIT_NO_REGULAR',
                    # 'FC_LOGINIT_NO_REGULAR_40',
                    # 'LOGINIT_NO_REGULAR',
                    # 'MIX_LOGINIT_L2_NO_REGULAR',
                    # 'MIX_LOGINIT_NO_REGULAR_L2'
                    ]:
        files = os.listdir('./LeNet/eval_efforts_rough/')
        files = filter(lambda a: "Constraint_mnist_GDBack_Compression_"+folder == '_'.join(a.split('_')[:-1]), files)
        models = map(lambda a: a.replace("Constraint_mnist_GDBack_Compression_"+folder,""), files)
        x = []
        y = []

        for model in models:
            print(folder+model)
            if os.path.exists('./LeNet/eval_efforts_rough/Constraint_mnist_GDBack_Compression_'+folder+model):
                x.append(float(model.split('_')[-1].split('.pkl')[0]))
                y.append(distortion_L2('./LeNet/eval_efforts_rough/Constraint_mnist_GDBack_Compression_'+folder+model)[0])
            else:
                print(folder+model+'not_found')

        # y = map(lambda a: a - y[-1],y)
        x,y = zip(*(sorted(zip(x,y))))
        ax0.plot(x,y,label=folder)
        x,y = get_accuracy_vs_NNsize('LeNet',folder)
        ax1.plot(x,y,label=folder)
    ax1.set_ylim([0.9,1])
    # plt.hlines(0,0,431080)
    plt.legend(loc=0)
    plt.gca().invert_xaxis()
    plt.show()
    exit()

if __name__ == '__main__':

    # models1 = [
    # "RandInit_Contract5_all_e0",
    # "RandInit_Contract_Like_e0_all",
    # "RandInit_Contract_Like_e2_all",
    # "RandInit_Contract_Like_e1_all",
    # "RandInit_Contract_Like_e3_all",
    # # "RandInit_FGS_0.1",
    # "RandInit_L2R_en3",
    # "RandInit_L2R_en2",
    #
    # "RandInit",
    # "RandInit_Distiliation_10",
    # "RandInit_Distiliation_20"
    #            ]

    # models1 = [
    # # "RandInit_Contract5_e0_L2_en3_all",
    # "RandInit_Contract_Like_e2_L2_en3_all",
    # # "RandInit_L2R_en3",
    # # "RandInit_L2R_en2",
    # "RandInit",
    # "Compression_431080.0"
    # ]
    #
    # for model in models1:
    #     plot_distortion('./LeNet/eval_efforts/Constraint_mnist_GDBack_'+model+'.pkl')
    # plt.show()
    # exit()


    # models = [
    #     ["_2080.0.pkl"],
    #     ["_4080.0.pkl"],
    #     ["_6080.0.pkl"],
    #     ["_8080.0.pkl"],
    #     ["_10080.0.pkl"],
    #     ["_12080.0.pkl"],
    #     ["_14080.0.pkl"],
    #     ["_16080.0.pkl"],
    #     ["_18080.0.pkl"],
    #     ["_20080.0.pkl"],
    #     ["_22080.0.pkl"],
    #     ["_24080.0.pkl"],
    #     ["_26080.0.pkl"],
    #     ["_28080.0.pkl"],
    #     ["_30080.0.pkl"],
    #     ["_32080.0.pkl"],
    #     ["_36080.0.pkl"],
    #     ["_38080.0.pkl"],
    #     ["_40080.0.pkl"],
    #     ["_51080.0.pkl"],
    #     ["_71080.0.pkl"],
    #     ["_91080.0.pkl"],
    #     ["_111080.0.pkl"],
    #     ["_151080.0.pkl"],
    #     ["_201080.0.pkl"],
    #     ["_251080.0.pkl"],
    #     ["_301080.0.pkl"],
    #     ["_351080.0.pkl"],
    #     ["_391080.0.pkl"],
    #     ["_431080.0.pkl"]
    # ]

    # Cifar()
    LeNet()