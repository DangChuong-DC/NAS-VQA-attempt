
import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from Properties.model import genotypes
from Properties.data.load_data import DataSet
from Properties.model.net import Net
from Properties.model.optim import get_net_optim, adjust_lr
from Properties.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval
from utils.accuracy import acc_eval


class Execution:
    def __init__(self, __T):
        self.__T = __T
        self.genotype = eval("genotypes.%s" % self.__T.ARCH_NAME)

        print('Loading training set ........')
        self.dataset = DataSet(__T)

        self.dataset_eval = None
        if __T.EVAL_EVERY_EPOCH:
            __T_eval = copy.deepcopy(__T)
            setattr(__T_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__T_eval)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        #print inherited gene
        print('Architecture is: %s' % str(self.genotype))
        print('---------')

        # Define the MCAN model
        net = Net(
            self.__T,
            self.genotype,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__T.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__T.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__T.RESUME:
            print(' ========== Resume training')

            if self.__T.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__T.CKPT_PATH
            else:
                path = self.__T.CKPTS_PATH + \
                       'ckpt_' + self.__T.CKPT_VERSION + \
                       '/epoch' + str(self.__T.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_net_optim(self.__T, net, data_size, ckpt['lr_base'], count=True)
            optim._step = int(data_size / self.__T.BATCH_SIZE * self.__T.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__T.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__T.VERSION) in os.listdir(self.__T.CKPTS_PATH):
                shutil.rmtree(self.__T.CKPTS_PATH + 'ckpt_' + self.__T.VERSION)

            os.mkdir(self.__T.CKPTS_PATH + 'ckpt_' + self.__T.VERSION)

            optim = get_net_optim(self.__T, net, data_size, count=True)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__T.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__T.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__T.NUM_WORKERS,
                pin_memory=self.__T.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__T.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__T.NUM_WORKERS,
                pin_memory=self.__T.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.__T.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__T.LOG_PATH +
                'log_train_' + self.__T.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__T.LR_DECAY_LIST:
                adjust_lr(optim, self.__T.LR_DECAY_R)

            # Externally shuffle
            if self.__T.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            for step, (
                    img_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):

                optim.zero_grad()

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                pred = net(
                    img_feat_iter,
                    ques_ix_iter
                )

                loss = loss_fn(pred, ans_iter)
                # loss /= self.__T.GRAD_ACCU_STEPS
                loss.backward()
                # loss_sum += loss.cpu().data.numpy() * self.__T.GRAD_ACCU_STEPS
                loss_sum += loss.cpu().data.numpy()

                with torch.no_grad():
                    accuracy_iter = acc_eval(pred.detach(), ans_iter)

                if self.__T.VERBOSE:
                    if dataset_eval is not None:
                        mode_str = self.__T.SPLIT['train'] + '->' + self.__T.SPLIT['val']
                    else:
                        mode_str = self.__T.SPLIT['train'] + '->' + self.__T.SPLIT['test']

                    print("\r[version %s][epoch %2d][step %4d/%4d][%s] acc: %.3f, loss: %.4f, lr: %.2e" % (
                        self.__T.VERSION,
                        epoch + 1,
                        step,
                        int(data_size / self.__T.BATCH_SIZE),
                        mode_str,
                        accuracy_iter,
                        # loss.cpu().data.numpy() / self.__T.SUB_BATCH_SIZE,
                        loss.cpu().data.numpy(),
                        optim._rate
                    ), end='        ')

                # Gradient norm clipping
                if self.__T.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__T.GRAD_NORM_CLIP
                    )

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__T.CKPTS_PATH +
                'ckpt_' + self.__T.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                self.__T.LOG_PATH +
                'log_train_' + self.__T.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
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
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__T.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__T.CKPT_PATH
        else:
            path = self.__T.CKPTS_PATH + \
                   'ckpt_' + self.__T.CKPT_VERSION + \
                   '/epoch' + str(self.__T.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')

        # Store the prediction list
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        net = Net(
            self.__T,
            self.genotype,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        if self.__T.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__T.DEVICES)

        net.load_state_dict(state_dict)

        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__T.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__T.NUM_WORKERS,
            pin_memory=True
        )

        for step, (
                img_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__T.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()

            pred = net(
                img_feat_iter,
                ques_ix_iter
            )
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            # Save the answer index
            if pred_argmax.shape[0] != self.__T.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__T.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            ans_ix_list.append(pred_argmax)

            # Save the whole prediction vector
            if self.__T.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__T.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__T.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )

                pred_list.append(pred_np)

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
                    self.__T.CACHE_PATH + \
                    'result_run_' + self.__T.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__T.CACHE_PATH + \
                    'result_run_' + self.__T.VERSION + \
                    '.json'

        else:
            if self.__T.CKPT_PATH is not None:
                result_eval_file = \
                    self.__T.RESULT_PATH + \
                    'result_run_' + self.__T.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__T.RESULT_PATH + \
                    'result_run_' + self.__T.CKPT_VERSION + \
                    '_epoch' + str(self.__T.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # Save the whole prediction vector
        if self.__T.TEST_SAVE_PRED:

            if self.__T.CKPT_PATH is not None:
                ensemble_file = \
                    self.__T.PRED_PATH + \
                    'result_run_' + self.__T.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__T.PRED_PATH + \
                    'result_run_' + self.__T.CKPT_VERSION + \
                    '_epoch' + str(self.__T.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


        # Run validation script
        if valid:
            # create vqa object and vqaRes object
            ques_file_path = self.__T.QUESTION_PATH['val']
            ans_file_path = self.__T.ANSWER_PATH['val']

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
                    self.__T.LOG_PATH +
                    'log_train_' + self.__T.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__T.LOG_PATH +
                    'log_train_' + self.__T.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('Write to log file: {}'.format(
                    self.__T.LOG_PATH +
                    'log_train_' + self.__T.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__T.LOG_PATH +
                    'log_train_' + self.__T.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.close()


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__T.VERSION)
            self.train(self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__T.LOG_PATH + 'log_train_' + version + '.txt')):
            os.remove(self.__T.LOG_PATH + 'log_train_' + version + '.txt')
        print('Finished!')
        print('')
