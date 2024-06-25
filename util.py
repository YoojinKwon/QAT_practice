import torch

class AverageMeter(object): # Compute and store average and current value
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_topk_accs(outputs, targets, topks=(1,)):
    with torch.no_grad():
        max_topk = max(topks)

        _, preds = outputs.topk(max_topk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets.view(1, -1).expand_as(preds))

        accs = []
        for topk in topks:
            acc = corrects[:topk].reshape(-1).float().sum(0, keepdim=True)
            accs.append(acc.mul_(100.0 / targets.size(0)))
        return accs
