import torch
import torch.nn as nn
import tensorly as tl
import sys
sys.path.append("../")



def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

class TKD2_layer(torch.nn.Module):
    '''
                      
    '''
    def __init__(self, layer, ranks, factors, num_groups=3):
        """
        """
        super(TKD2_layer, self).__init__()

        self.ranks = ranks
#         if split:
#             self.cin = self.data_split()
        self.cin = layer.in_channels
        self.cout = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.kernel_size = layer.kernel_size
        self.is_bias = layer.bias is not None
        self.num_groups = num_groups
        if self.is_bias:
            self.bias = layer.bias
            
        self.tucker_decomposition = self.__replace__(factors)


    def __replace__(self, factors):
        """ Gets a layer factoriztion, 
            returns a nn.Sequential object with the Tucker decomposition.
        """
        core, [last, first] = factors

        self.first_layer = torch.nn.Conv2d(in_channels=self.cin, \
                out_channels=self.ranks[0], kernel_size = (1, 1),
                stride=1, dilation=self.dilation, bias=False)
        

        self.core_layer = torch.nn.Conv2d(in_channels=self.ranks[0], \
                out_channels=self.ranks[1], kernel_size=self.kernel_size,
                stride=1, padding=self.padding, dilation=self.dilation,
                bias=False)

        self.last_layer = torch.nn.Conv2d(in_channels=self.ranks[1], \
            out_channels=self.cout, kernel_size = (1, 1), stride=1, bias=True)
        if self.is_bias:
            self.last_layer.bias.data = self.bias.data

        self.first_layer.weight.data = \
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        
        self.last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        self.core_layer.weight.data = core
       
        in_splits = _split_channels(self.cin, self.num_groups)

        ch_out_lst = [self.cin, self.ranks[0], self.ranks[1]]

        linear = []
        for idx, in_ch in enumerate(in_splits):
            linear.append(torch.nn.Linear(in_ch, ch_out_lst[idx]))
        
        self.linear3 = torch.nn.Linear(self.ranks[0] + self.ranks[1] + self.cout, self.cout)
        # self.last = torch.nn.Conv2d(self.ranks[0] + self.ranks[1] + self.cout, self.cout, 1)
        
        
        self.linears = nn.ModuleList(linear)
        self.splits = in_splits



    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = []
        # print(self.ranks)
        for spx, linears in zip(x_split, self.linears):
            x_out.append(linears(spx.permute(0,3,2,1)).permute(0,3,2,1))
        x1 = self.first_layer(x_out[0])
        # x2 = self.last_layer(self.linear1(x.permute(0,3,2,1)).permute(0,3,2,1))
        x2 = self.last_layer(x_out[1])

        # x3 = self.core_layer(self.linear2(x.permute(0,3,2,1)).permute(0,3,2,1))
        x3 = self.core_layer(x_out[2])

        catr = self.linear3(torch.cat((x1, x2, x3), dim=1).permute(0,3,2,1)).permute(0,3,2,1)
        # catr = self.linear3(torch.cat((x1, x2, x3), dim=1)) 


        return catr
    
    
    