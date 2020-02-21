import os
import torch
import tabulate
import torch.nn as nn

class Hook_record_input():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input

    def close(self):
        self.hook.remove()
        
def add_input_record_Hook(model, name_as_key=False):
    Hooks = {}
    if name_as_key:
        for name,module in model.named_modules():
            #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            Hooks[name] = Hook_record_input(module)
            
    else:
        for k,module in enumerate(model.modules()):
            #if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            Hooks[k] = Hook_record_input(module)
    return Hooks

def remove_hooks(Hooks):
    for k in Hooks.keys():
        Hooks[k].close()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def run_epoch(loader, model, criterion, optimizer=None,
              phase="train", loss_scaling=1.0):
    assert phase in ["train", "val", "test"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="val" or phase=="test": model.eval()

    ttl = 0
    Hooks = add_input_record_Hook(model)
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase=="train":
                optimizer.zero_grad()
                loss = loss * loss_scaling # grad scaling
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
    }

