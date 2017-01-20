from LeNet_mask import *

if __name__ == '__main__':

    theano.sandbox.cuda.use("gpu"+sys.argv[1])

    folders = [
        # 'OBD_L2',
        # 'OBD_LOGINIT_L2',
        # 'OBD_LOGINIT_NO_REGULAR',

        'NO_REGULAR',
        'LOGINIT_NO_REGULAR',
        # 'LOGINIT_NO_REGULAR_KEEPBIAS',
        # 'NO_REGULAR_RANDOMINIT1',
        # 'NO_REGULAR_DROPOUT_0.2',
        # 'NO_REGULAR_DROPOUT_0.5',
        # 'NO_REGULAR_DROPOUT_0.9',
        # 'NO_REGULAR_RANDOM1',
        # 'NO_REGULAR_RANDOM2',
        # 'NO_REGULAR_RANDOM3',
        # 'NO_REGULAR_RANDOM4',
        # 'LOGINIT_NO_REGULAR_RANDOMINIT1',
        # 'CC_LOGINIT_NO_REGULAR',
        # 'CC_NO_REGULAR',
        'CCV3_NO_REGULAR',
        # 'CCV3_0.05_NO_REGULAR'
        'CCV3_LOGINIT_NO_REGULAR',
        # 'CCV2_LBL_LOGINIT_NO_REGULAR',
        # 'CCV2_LBL_LOGINIT_NO_REGULAR'
        'L2EN3',
        'LOGINIT_L2EN3',
        # 'LOGINIT_L2_KEEPBIAS',
        'CCV3_L2',
        'CCV3_LOGINIT_L2',
        'CONTRACT_LIKE',
        'LOGINIT_CONTRACT_LIKE',
        # 'LOGINIT_CONTRACT_LIKE_KEEPBIAS',
        'CCV3_CONTRACT_LIKE',
        'CCV3_LOGINIT_CONTRACT_LIKE',
        'FC_LOGINIT_CONTRACT_LIKE_L2MASK',
        'FC_LOGINIT_NO_REGULAR_L2MASK',
        'FC_LOGINIT_NO_REGULAR_INITL2MASK',
        'FC_LOGINIT_L2_INITCONTRACTLIKEMASK',
        'FC_LOGINIT_NO_REGULAR_INITCONTRACTLIKEMASK',
        'FC_LOGINIT_L2_INITNOREGULARMASK',
        'FC_LOGINIT_CONTRACT_LIKE_INITNOREGULARMASK',
        'FC_LOGINIT_CONTRACT_LIKE_INITL2MASK',

        # 'CCV2_LBL_LOGINIT_NO_REGULAR',

        # 'SWCC_LOGINIT_NO_REGULAR',
        # 'LOGINIT_NO_REGULAR_0.05',
        #  'LOGINIT_NO_REGULAR_0.15'
        #  'CONTRACT_LIKE',
        # 'CONTRACT_LIKE_L2',
        # 'LOGINIT_L2EN3',
        # 'L1',
        # 'LOGINIT_L1',

        # 'CC_NO_REGULAR',
        # 'CC_LOGINIT_NO_REGULAR',
        # 'LOGINIT_CONTRACT_LIKE_L2',
        # 'FC_LOGINIT_CONTRACT_LIKE_L2_L2MASK',
        # 'FC_CONTRACT_LIKE_L2MASK',
        # 'FC_LOGINIT_CONTRACT_LIKE',
        # 'CC_CONTRACT_LIKE'
        # 'LBL_LOGINIT_NO_REGULAR',
        # 'PCC_LOGINIT_NO_REGULAR',
        # 'FC_LOGINIT_NO_REGULAR_40',
        # 'MIX_LOGINIT_L2_NO_REGULAR',
        # 'MIX_LOGINIT_NO_REGULAR_L2'
        ]

    for folder in folders[int(sys.argv[2]):int(sys.argv[3])]:
        files = os.listdir('./Compression/' + folder + '/')
        files = filter(lambda x: 'connection_count' in x, files)
        print files
        files = [int(i.split("_")[2].split('.')[0]) for i in files]
        files.sort(reverse=True)
        files = ["connection_count_"+str(i)+'.0.pkl' for i in files]
        eval_adversarial_efforts(files, folder, dropout_rate=-1)
