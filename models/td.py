import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_TD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Conv2d_TD, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size
    
    def forward(self, input):
        # sort blocks by mean absolute value
        block_values = F.avg_pool2d(self.weight.data.abs().permute(2,3,0,1),
                        kernel_size=(self.block_size, self.block_size),
                        stride=(self.block_size, self.block_size))
        sorted_block_values, indices = torch.sort(block_values.contiguous().view(-1))
        thre_index = int(block_values.data.numel() * self.gamma)
        threshold = sorted_block_values[thre_index]
        mask_small = 1 - block_values.gt(threshold.cuda()).float().cuda() # mask for blocks candidates for pruning
        mask_dropout = torch.rand_like(block_values).lt(self.alpha).float().cuda()
        mask_keep = 1.0 - mask_small * mask_dropout
        mask_keep_original = F.interpolate(mask_keep, 
                            scale_factor=(self.block_size, self.block_size)).permute(2,3,0,1)
        out = F.conv2d(input, self.weight * mask_keep_original, None, self.stride, self.padding,
                                    self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class Conv2d_col_TD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Conv2d_col_TD, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size

    def forward(self, input):
        w_l = self.weight
        num_group = w_l.size(0) * w_l.size(1) // self.block_size
        w_i = w_l.view(self.block_size * w_l.size(2) * w_l.size(3), num_group)  # reshape the weight tensor into 2-D matrix
    
        w_norm = w_i.norm(p=2, dim=0)
        sorted_col, indices = torch.sort(w_norm.contiguous().view(-1), dim=0)
        th_idx = int(w_norm.numel() * self.gamma)
        threshold = sorted_col[th_idx]
    

        mask_small = 1 - w_norm.gt(threshold).float()
        mask_dropout = torch.rand_like(w_norm).lt(self.alpha).float()
        mask_keep = 1 - mask_small * mask_dropout
    
        mask_keep_2d = mask_keep.expand(w_i.size())
        mask_keep_original = mask_keep_2d.resize_(w_l.size())

        out = F.conv2d(input, self.weight * mask_keep_original, None, self.stride, self.padding,
                                    self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


        

class Linear_TD(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Linear_TD, self).__init__(in_features, out_features, bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size

    def forward(self, input):
        block_values = F.avg_pool2d(self.weight.data.abs().unsqueeze(0),
                        kernel_size=(self.block_size, self.block_size),
                        stride=(self.block_size, self.block_size))
        sorted_block_values, indices = torch.sort(block_values.contiguous().view(-1))
        thre_index = int(block_values.data.numel() * self.gamma)
        threshold = sorted_block_values[thre_index]
        mask_small = 1 - block_values.gt(threshold.cuda()).float().cuda() # mask for blocks candidates for pruning
        mask_dropout = torch.rand_like(block_values).lt(self.alpha).float().cuda()
        mask_keep = 1.0 - mask_small * mask_dropout
        mask_keep_original = F.interpolate(mask_keep.unsqueeze(0), 
                            scale_factor=(self.block_size, self.block_size)).squeeze()
        return F.linear(input, self.weight * mask_keep_original, self.bias)

