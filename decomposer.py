import torch
import tensorly as tl
from tensorly.tucker_tensor import tucker_to_tensor
import torch
from math import log10, sqrt, log2

tl.set_backend('pytorch')

class decompose():
    def __init__(self, 
                tensor, 
                factors, 
                n_iter=5000,
                lr=0.0005,
                penalty=0.1,
                print_iter = 1000,
                **kwargs
                 ):
        
        self.n_iter = n_iter
        self.lr = lr
        self.penalty = penalty
        self.tensor = tensor
        self.ori_mean = torch.mean(tensor)
        self.core = kwargs.get('core', None)
        self.factors = factors
        self.optimizer = kwargs.get('optimizer', "local")
        self.ftype = kwargs.get('ftype', None)
        self.print_iter = print_iter
#         self.optim = None

        
     
        if self.optimizer.lower() =="adam":
            if self.ftype == 'mf':
                self.optim = torch.optim.Adam([self.factors], lr=lr)
            else:
                self.optim = torch.optim.Adam([self.core] + self.factors, lr=lr)
           
        elif self.optimizer.lower() == "sgd":
            if self.ftype == 'mf':
                self.optim = torch.optim.SGD([self.factors], lr=lr)
            else:
                self.optim = torch.optim.SGD([self.core] + self.factors, lr=lr)
           
 
    def SGD(self, params, lr):
        for param in params:
            param[:] = param - lr * param.grad
  
    def train(self):
        if self.ftype == None or self.ftype == "tucker2":
            
            print("Tucker Decomposing...")
        else:
            print("Matrix Decomposing...")
        
        print("Optimizer:", self.optimizer) 

        for i in range(1, self.n_iter):
            fact_distance = 0
            core_dist = 0

            # Important: do not forget to reset the gradients
            if self.optimizer.lower() == "ADAM" or self.optimizer.lower() == "SGD":
                self.optim.zero_grad()
            
            elif self.optimizer.lower() == "Local":
                if self.ftype == "mf":
                    for param in self.factors:
                        param.grad = None
                else:   
                    for param in [self.core] + self.factors:
                        param.grad = None
            
             # Reconstruct the tensor from the decomposed form
            if self.ftype == "mf":
                self.rec = torch.matmul(self.factors[0], self.factors[1].T)
            else:
                self.rec = tucker_to_tensor(self.core, self.factors)

             # squared l2 loss
            loss = (self.rec - self.tensor).pow(2).sum()
            
            if self.ftype == "mf":
                difference = sum((factor.var() - self.tensor.var()) for factor in self.factors) 
                loss = loss + self.penalty * difference
            else:
                difference = sum((factor.var() - self.tensor.var()) for factor in self.factors) + (self.core.var() - self.tensor.var())  
                
                loss = loss + self.penalty * difference

             # squared penalty on the factors of the decomposition
#             if self.ftype == "mf":
#                 for lists in self.factors:
#                     for numbers in lists:
#                         for number in numbers:
#                             fact_distance += (abs(number.item()) - self.ori_mean) ** 2
#                             loss = loss + self.penalty * fact_distance 
#             else:
#                 for elements in self.factors:
#                     for items in elements:
#                         for values in items:
#                             fact_distance += (abs(values.item()) - self.ori_mean)**2
#                             loss = loss + self.penalty * fact_distance
#                 if self.core.ndim == 3:
#                     for ele in self.core:
#                         for itm in ele:
#                             for vals in itm:
#                                 core_dist += (abs(vals.item()) - self.ori_mean)**2
#                                 loss = loss + self.penalty * core_dist
#                 elif self.core.ndim == 2:
#                     for ele in self.core:
#                         for itm in ele:
#                             core_dist += (abs(itm.item()) - self.ori_mean)**2
#                             loss = loss + self.penalty * core_dist
#                 else:
#                     for ele in self.core:
#                         for itm in ele:
#                             for vals in itm:
#                                 for nums in vals:
#                                     core_dist += (abs(nums.item()) - self.ori_mean)**2
#                                     loss = loss + self.penalty * core_dist

       
            loss.backward(retain_graph=False)
         
            with torch.no_grad():
                if self.optimizer.lower() == "ADAM" or self.optimizer.lower() == "SGD":
                    self.optim.step()
                elif self.optimizer.lower() == "local":
                    if self.ftype == 'mf':
                        self.SGD(self.factors, self.lr)
                    else:
                        self.SGD([self.core] + self.factors, self.lr)

            if i % self.print_iter == self.print_iter - 1:
                self.rec_error = tl.norm(self.rec - self.tensor, 2)/tl.norm(self.tensor, 2)
                print("Epoch %s,. Rec. error: %s, Variance %s" % (i, self.rec_error, difference))