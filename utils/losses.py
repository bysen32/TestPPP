import torch
import torch.nn.functional as F
from torch.autograd import Variable


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, predict, target):
        pt = torch.sigmoid(predict)

        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
                - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1)).cuda()
        else:
            self.alpha = alpha.cuda()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predict, target):
        # softmax 获取预测概率
        pt = F.softmax(predict, dim=1)
        # 获取target的one hot编码
        class_mask = F.one_hot(target, self.num_classes)
        ids = target.view(-1, 1)
        # 注意，这里的alpha是给定的一个list(tensor #) 里面的元素分别是一个类的权重因子
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1)
        # 利用onehot作为mask, 提取对应的pt
        probs = (pt * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Crossentropy_LabelSmoothing(torch.nn.Module):
    '''
    NLL loss with label smoothing
    '''
    def __init__(self, smoothing=0.0):
        '''
        Constructor for the labelsmoothing module.
        :param smoothing: label smoothing factor
        '''
        super(Crossentropy_LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()

def get_criterion(args):
    if args.focal_loss:
        criterion = MultiCEFocalLoss(num_classes=args.num_classes).cuda()
    elif args.label_smoothing != 0:
        criterion = Crossentropy_LabelSmoothing(smoothing=args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    return criterion
