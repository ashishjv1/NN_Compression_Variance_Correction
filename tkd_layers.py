import torch
import torch.nn as nn
import tensorly as tl
import sys
sys.path.append("../")


class TKD2_layer(torch.nn.Module):
    '''
                      
    '''
    def __init__(self, layer, ranks, factors, split=True):
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
            last_layer.bias.data = self.bias.data

        self.first_layer.weight.data = \
            torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        
        self.last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        self.core_layer.weight.data = core
       
        
        self.linear1 = torch.nn.Linear(self.cin, self.ranks[0])
        self.linear2 = torch.nn.Linear(self.cin, self.ranks[1])
        
        self.linear3 = torch.nn.Linear(self.ranks[0] + self.ranks[1] + self.cout, self.cout)
     
    
    
    
    
    ######Need to think Logic Here######
    def data_split(self, x, divisor=3, no_split=2):
        
        shape = x.shape[1]
        if no_split == 3:
            if shape % divisor == 0:
                a = x[:,0:int(shape/divisor),:,:]
                b = x[:,int(shape/divisor): 2 * int(shape/divisor),:,:]
                c = x[:,2 * int(shape/divisor): 3*int(shape/divisor),:,:]
                
            elif shape % divisor >= 1:
                rem = shape % divisor
                a = x[:,0:int(shape/divisor) + rem,:,:]
                b = x[:,int(shape/divisor) + rem : 2 * int(shape/divisor) + rem,:,:]
                c = x[:,2 * int(shape/divisor) + rem : 3 * int(shape/divisor) + rem,:,:]
            
            return a,b,c
        
        
        elif no_split == 2:
            divisor = 2
            if shape % divisor == 0:
                a = x[:,0:int(shape/divisor),:,:]
                b = x[:,int(shape/divisor): 2 * int(shape/divisor),:,:]
              
            elif shape % divisor >= 1:
                rem = shape % divisor
                a = x[:,0:int(shape/divisor) + rem,:,:]
                b = x[:,int(shape/divisor) + rem : 2 * int(shape/divisor) + rem,:,:]
        
            return a,b
       

    def forward(self, x):
        
        
        x1 = self.first_layer(x)

        l1 = self.linear1(x.permute(0,3,2,1)).permute(0,3,2,1)

        x2 = self.last_layer(l1)
        
        x3 = self.core_layer(self.linear2(x.permute(0,3,2,1)).permute(0,3,2,1))

        catr = self.linear3(torch.cat((x1, x2, x3), dim=1).permute(0,3,2,1)).permute(0,3,2,1) 

        return catr
    
    
    