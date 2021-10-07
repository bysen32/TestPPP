import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

def get_optimizer(optim_name, model, args):
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise('no {}'.format(optim_name))
    return optimizer

def get_scheduler(sched_name, optimizer, args):
    if sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif sched_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, verbose=True)
    else:
        raise('no {}'.format(sched_name))

    return scheduler

class WarmUpLR(_LRScheduler):
    '''
    warmup_training learning rate scheduler
    Args:
        optimizer:
        total_iters: total_iters of warmup phase
    '''
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        '''
        We will use the first m batches, and set the learning rate
        to base_lr * m / total_iters
        '''
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
