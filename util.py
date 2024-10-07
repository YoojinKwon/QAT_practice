import torch
import matplotlib.pyplot as plt

def visulaize_before_after_quantizer(fp, q): 
    # Assuming fp_act and q_act are PyTorch tensors
    x1 = fp_act.numpy().flatten()
    x2 = q_act.numpy().flatten()

    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Custom colors
    colors = ['#FF7F0E', '#1F77B4']  # Orange and Blue

    # Create the first histogram with custom color
    axes[0].hist(x1, bins=100, color=colors[0], edgecolor='black', alpha=0.7)
    axes[0].set_title('Histogram of fp_activation', fontsize=14, fontweight='bold', color='#2F4F4F')
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    # Create the second histogram with custom color
    axes[1].hist(x2, bins=100, color=colors[1], edgecolor='black', alpha=0.7)
    axes[1].set_title('Histogram of q_activation', fontsize=14, fontweight='bold', color='#2F4F4F')
    axes[1].set_xlabel('Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the histograms
    plt.show()
    
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

def model_train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, running_train_acc, running_train_loss):
    model.train()
    running_train_loss.reset()
    running_train_acc.reset()

    progress_bar = tqdm(train_loader)
    
    for step, (imgs, labels) in enumerate(progress_bar):
        imgs, labels = imgs.to(device), labels.to(device) # Images: [batch, 3, 32, 32]
        
        optimizer.zero_grad() # every step, zero grad        
        outputs = model(imgs) # forward
        
        loss = criterion(outputs, labels) # get loss
        accs = calculate_topk_accs(outputs, labels) # get acc
        running_train_loss.update(loss.item(), imgs.size(0))
        running_train_acc.update(accs[0].item(), imgs.size(0))

        loss.backward() # backward
        optimizer.step() # tune parameters
  
    return running_train_loss.avg, running_train_acc.avg

def model_validate_one_epoch(epoch, model, val_loader, criterion, device, running_val_acc, running_val_loss):    
    model.eval()
    running_val_loss.reset()
    running_val_acc.reset()

    progress_bar = tqdm(val_loader)
    with torch.no_grad():
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            accs = calculate_topk_accs(outputs, labels)
            running_val_loss.update(loss.item(), imgs.size(0))
            running_val_acc.update(accs[0].item(), imgs.size(0))            
    
    return running_val_loss.avg, running_val_acc.avg
