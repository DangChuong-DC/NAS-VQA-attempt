
import torch, math
import torch.optim as Optim


class Optimizer(object):
    def __init__(self, lr_base, eta_min, optimizer, max_epoch, data_size, batch_size, search=False):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.eta_min = eta_min
        self.T_max = max_epoch
        self.search = search


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if self.search:
            r = self.lr_base
        else:
            if step is None:
                step = self._step
            step_epo = int(self.data_size / self.batch_size)
            if step <= step_epo * 1:
                r = self.lr_base * 1/4.
            elif step <= step_epo * 2:
                r = self.lr_base * 2/4.
            elif step <= step_epo * 3:
                r = self.lr_base * 3/4.
            else:
                r = self.lr_base

        return r


def get_net_optim(__C, model, data_size, lr_base=None, count=False, search=False):
    if lr_base is None:
        lr_base = __C.LR_BASE
    if __C.N_GPU > 1:
        net_params = model.module.net_parameters()
    else:
        net_params = model.net_parameters()

    if count:
        print(">>> Number of parameters: %d\n" % sum(p.numel() for p in net_params))

    return Optimizer(
        lr_base,
        __C.LR_MIN,
        Optim.Adam(
            net_params,
            lr=0,
            betas=__C.OPT_BETAS,
            eps=__C.OPT_EPS
        ),
        __C.MAX_EPOCH,
        data_size,
        __C.BATCH_SIZE,
        search=search,
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r


def get_arch_optim(__C, model):
    if __C.N_GPU > 1:
        arch_params = model.module.arch_parameters()
    else:
        arch_params = model.arch_parameters()

    return Optim.Adam(
        arch_params,
        __C.ARCH_LR,
        betas=__C.ARCH_BETAS,
        eps=__C.ARCH_EPS,
        weight_decay=__C.ARCH_W_DECAY,
    )
