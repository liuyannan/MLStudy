from LeNet_mask import *

if __name__ == '__main__':

    theano.sandbox.cuda.use("gpu"+sys.argv[1])
    # for folder in [
    #     'CC_LOGINIT_NO_REGULAR',
    #     'CC_NO_REGULAR',
    #     'CCV2_LBL_LOGINIT_NO_REGULAR'
    # ]:
    #     eval_contractive_term(folder)
    #
    # exit()

    folders = [
               # ['FC_LOGINIT_NO_REGULAR_INITCONTRACTLIKEMASK', 'no_regular', 'SW', 'LOGINIT_CONTRACT_LIKE'],
               # ['FC_LOGINIT_NO_REGULAR_INITL2MASK', 'no_regular', 'SW', 'LOGINIT_L2EN3'],
               # ['FC_L2_INITNOREGULARMASK', 'L2', '', 'LOGINIT_NO_REGULAR'],
               # ['FC_L2_INITCONTRACTLIKEMASK', 'L2', '', 'LOGINIT_CONTRACT_LIKE'],
               # ['FC_LOGINIT_CONTRACT_LIKE_INITL2MASK', 'Contract_Likelihood', '', 'LOGINIT_L2EN3'],
        # ['LOGINIT_CONTRACT_LIKE_KEEPBIAS', 'Contract_Likelihood', 'SW', ''],
        # ['LOGINIT_L2_KEEPBIAS', 'L2', 'SW', ''],
        # ['CCV3_L2', 'L2', 'CCV3', ''],
        # ['CCV3_LOGINIT_L2', 'L2', 'CCV3', ''],
        # ['CCV3_CONTRACT_LIKE', 'Contract_Likelikhood', 'CCV3', ''],
        # ['CCV3_LOGINIT_CONTRACT_LIKE', 'Contract_Likelihood', 'CCV3', ''],
        ['RANDOM_LOGINIT_NO_REGULAR_1', 'no_regular', 'RANDOM', -1],
        ['RANDOM_LOGINIT_NO_REGULAR_2', 'no_regular', 'RANDOM', -1],
        ['RANDOM_LOGINIT_NO_REGULAR_3', 'no_regular', 'RANDOM', -1],
        ]
    for folder in [folders[int(sys.argv[2])]]:
        if not os.path.exists('./Compression/' + folder[0]):
            os.mkdir('./Compression/' + folder[0])
        # fixed_compression(mask_folder=folder[3], target_floder=folder[0], target_cf=folder[1])
        compression_API(folder[0], cf=folder[1], rm_policy= folder[2],resume=False, ratio=0.1, dropout_rate=folder[3])
        # compression(folder[0],folder[1])
        # SW_CC_compression(folder=folder[0],cf=folder[1],ratio=0.1)
        files = os.listdir('./Compression/' + folder[0] + '/')
        files = [int(i.split("_")[2].split('.')[0]) for i in files]
        files.sort(reverse=True)
        files = ["connection_count_"+str(i)+'.0.pkl' for i in files]
        eval_adversarial_efforts(files, folder[0],dropout_rate=folder[3])
        eval_accuracy(folder[0],dropout_rate=folder[3])
        eval_contractive_term(folder[0],dropout_rate=folder[3])