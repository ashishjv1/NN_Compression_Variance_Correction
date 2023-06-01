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
                optimizer=None,
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
        self.optimizer = optimizer
        self.ftype = kwargs.get('ftype', None)
        self.print_iter = print_iter

        if optimizer:
           if optimizer.lower() =="ADAM":
              self.optimizer = torch.optim.Adam([self.core] + self.factors, lr=lr) # type: ignore
           elif optimizer.lower() == "SGD":
              self.optimizer = torch.optim.SGD([self.core]+ self.factors, lr=lr) # type: ignore

 
    def SGD(self, params, lr):
        for param in params:
            param[:] = param - lr * param.grad
  
    def train(self):
      if self.ftype == None:
         
         print("Tucker Decomposing...")
      else:
         print("Matrix Decomposing...")

      for i in range(1, self.n_iter):
         fact_distance = 0
         core_dist = 0

         # Important: do not forget to reset the gradients
         if self.optimizer:
            self.optimizer.zero_grad()
         elif self.ftype == "mf":
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

         # squared penalty on the factors of the decomposition
         if self.ftype == "mf":
            for lists in self.factors:
               for numbers in lists:
                  for number in numbers:
                     fact_distance += (abs(number.item()) - self.ori_mean) ** 2
                     loss = loss + self.penalty * fact_distance 
         else:
            for elements in self.factors:
               for items in elements:
                  for values in items:
                     fact_distance += (abs(values.item()) - self.ori_mean)**2
                     loss = loss + self.penalty * fact_distance
            if self.core.ndim == 3:
               for ele in self.core:
                  for itm in ele:
                     for vals in itm:
                        core_dist += (abs(vals.item()) - self.ori_mean)**2
                        loss = loss + self.penalty * core_dist
            elif self.core.ndim == 2:
               for ele in self.core:
                  for itm in ele:
                     core_dist += (abs(itm.item()) - self.ori_mean)**2
                     loss = loss + self.penalty * core_dist
            else:
               for ele in self.core:
                  for itm in ele:
                     for vals in itm:
                        for nums in vals:
                           core_dist += (abs(nums.item()) - self.ori_mean)**2
                           loss = loss + self.penalty * core_dist
       
         loss.backward(retain_graph=False)
         
         with torch.no_grad():
            if self.optimizer: 
               self.optimizer.step()
            else:
               if self.ftype == 'mf':
                  self.SGD(self.factors, self.lr)
               else:
                  self.SGD([self.core] + self.factors, self.lr)

         if i % self.print_iter == 0:
            self.rec_error = tl.norm(self.rec - self.tensor, 2)/tl.norm(self.tensor, 2)
            print("Epoch %s,. Rec. error: %s, Variance %s" % (i, self.rec_error, fact_distance))