import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
class FN(nn.Module):
    def	__init__(self):
        super(FN, self).__init__()
        self.input_shape = 1024
        self.num_actions = 10
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, self.num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return x


a = torch.rand(1024, 1024)

for i in range(1, 9):
    torch.set_num_threads(i) 
    print('number of threads used: ', torch.get_num_threads())
    tic = timeit.default_timer()
    net= FN()
    for i in range(2500): 
        y = net(a)
    toc = timeit.default_timer() 
    print(toc - tic)