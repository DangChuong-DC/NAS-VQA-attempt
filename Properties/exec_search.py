
import os, json, torch, datetime, pickle, copy, shutil, time, math
import numpy as np
import torch.nn as nn
import torch.utils.data as Data


from Properties.data.load_data import DataSet
from Properties.model.net_search import Net_Search
from Properties.model.optim import get_net_optim, get_arch_optim, adjust_lr
from Properties.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval
from utils.accuracy import acc_eval


class Search_Execution:
    def __init__(self, __S):
        self.__S = __S

        print('Loading training set ........')
        self.dataset = DataSet(__S)

        self.dataset_eval = None
        if __S.EVAL_EVERY_EPOCH:
            __S_eval = copy.deepcopy(__S)
            setattr(__S_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__S_eval)


    def search(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the model
        net = Net_Search(
            self.__S,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__S.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__S.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__S.RESUME:
            print(' ========== Resume training')

            if self.__S.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__S.CKPT_PATH
            else:
                path = self.__S.CKPTS_PATH + \
                       'ckpt_' + self.__S.CKPT_VERSION + \
                       '/epoch' + str(self.__S.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            arch_optim = get_arch_optim(self.__S, net)
            net_optim = get_net_optim(self.__S, net, data_size, ckpt['net_lr_base'],
                search=True)
            net_optim._step = int(data_size / self.__S.BATCH_SIZE * self.__S.CKPT_EPOCH)
            net_optim.optimizer.load_state_dict(ckpt['net_optimizer'])
            arch_optim.load_state_dict(ckpt['arch_optimizer'])

            start_epoch = self.__S.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__S.VERSION) in os.listdir(self.__S.CKPTS_PATH):
                shutil.rmtree(self.__S.CKPTS_PATH + 'ckpt_' + self.__S.VERSION)

            os.mkdir(self.__S.CKPTS_PATH + 'ckpt_' + self.__S.VERSION)

            arch_optim = get_arch_optim(self.__S, net)
            net_optim = get_net_optim(self.__S, net, data_size, search=True)
            start_epoch = 0

        loss_sum = 0
        # Obtain list of network & architecture parameters
        if self.__S.N_GPU > 1:
            net_params = net.module.net_parameters()
            arch_params = net.module.arch_parameters()
        else:
            net_params = net.net_parameters()
            arch_params = net.arch_parameters()


        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__S.BATCH_SIZE,
            shuffle=True,
            num_workers=self.__S.NUM_WORKERS,
            pin_memory=self.__S.PIN_MEM,
            drop_last=True
        )

        # Training script
        for epoch in range(start_epoch, self.__S.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__S.LOG_PATH +
                'log_search_' + self.__S.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            # if epoch in self.__S.LR_DECAY_LIST:
            #     adjust_lr(net_optim, self.__S.LR_DECAY_R)

            time_start = time.time()
            # Iteration
            for step, (img_feat_iter, ques_ix_iter, ans_iter) in enumerate(dataloader):
                # Zero gradient for optimizers
                net_optim.zero_grad()
                arch_optim.zero_grad()
                # Prepare data for training network parameters
                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()


                pred = net(
                    img_feat_iter,
                    ques_ix_iter
                )

                loss = loss_fn(pred, ans_iter)
                # loss = loss.mean()
                loss.backward()

                loss_sum += loss.cpu().data.numpy()

                with torch.no_grad():
                    accuracy_iter = acc_eval(pred.detach(), ans_iter)

                if self.__S.VERBOSE:
                    mode_str = 'search'

                    print("\r[version %s][epoch %2d][step %4d/%4d][%s] acc: %.3f, loss: %.4f, lr: %.2e, arch_lr: %.2e" % (
                        self.__S.VERSION,
                        epoch + 1,
                        step,
                        int(data_size / self.__S.BATCH_SIZE),
                        mode_str,
                        accuracy_iter,
                        loss.cpu().data.numpy(),
                        net_optim._rate,
                        self.get_opt_lr(arch_optim),
                    ), end='   ')

                # Gradient norm clipping
                if self.__S.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net_params,
                        self.__S.GRAD_NORM_CLIP
                    )
                net_optim.step()

                if self.__S.ARCH_GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        arch_params,
                        self.__S.ARCH_GRAD_NORM_CLIP
                    )
                arch_optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Get current genotype
            if self.__S.N_GPU > 1:
                best_gene = net.module.genotype()
            else:
                best_gene = net.genotype()

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'net_optimizer': net_optim.optimizer.state_dict(),
                'net_lr_base': net_optim.lr_base,
                'arch_optimizer': arch_optim.state_dict(),
            }
            torch.save(
                state,
                self.__S.CKPTS_PATH +
                'ckpt_' + self.__S.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__S.LOG_PATH +
                'log_search_' + self.__S.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(net_optim._rate) +
                '\n' +
                'current gene: ' + str(best_gene) +
                '\n\n'
            )
            logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )


            loss_sum = 0


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__S.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__S.CKPT_PATH
        else:
            path = self.__S.CKPTS_PATH + \
                   'ckpt_' + self.__S.CKPT_VERSION + \
                   '/epoch' + str(self.__S.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        # pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net_Search(
            self.__S,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        if self.__S.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__S.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__S.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__S.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):

            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__S.EVAL_BATCH_SIZE),
            ), end='   ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            pred = net(
                img_feat_iter,
                ques_ix_iter,
            )
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__S.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__S.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)


        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        # Write the results to result file
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__S.CACHE_PATH + \
                    'result_run_' + self.__S.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__S.CACHE_PATH + \
                    'result_run_' + self.__S.VERSION + \
                    '.json'

        else:
            if self.__S.CKPT_PATH is not None:
                result_eval_file = \
                    self.__S.RESULT_PATH + \
                    'result_run_' + self.__S.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__S.RESULT_PATH + \
                    'result_run_' + self.__S.CKPT_VERSION + \
                    '_epoch' + str(self.__S.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.__S.QUESTION_PATH['val']
            ans_file_path = self.__S.ANSWER_PATH['val']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print("\n")
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # print("Per Question Type Accuracy is the following:")
            # for quesType in vqaEval.accuracy['perQuestionType']:
            #     print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            # print("\n")
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")

            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.__S.LOG_PATH +
                    'log_search_' + self.__S.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__S.LOG_PATH +
                    'log_search_' + self.__S.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.__S.LOG_PATH +
                    'log_search_' + self.__S.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__S.LOG_PATH +
                    'log_search_' + self.__S.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n---------")
            logfile.write("\n\n")
            logfile.close()


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__S.VERSION)
            self.search(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)


        else:
            exit(-1)

    def get_opt_lr(self, arch_optim):
        for par_gr in arch_optim.param_groups:
            return par_gr['lr']

    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__S.LOG_PATH + 'log_search_' + version + '.txt')):
            os.remove(self.__S.LOG_PATH + 'log_search_' + version + '.txt')
        print('Finished!')
        print('')
