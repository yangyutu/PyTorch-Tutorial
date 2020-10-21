"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NetOneLayer(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NetOneLayer, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class NetMultiLayer(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(NetMultiLayer, self).__init__()
        if(len(n_hidden) < 1):
            raise ValueError("number of hidden layer should be greater than 0")
        # here we need to use nn.ModuleList in order to build a list of layers. 
        # we cannot use ordinary list
        self.layers = torch.nn.ModuleList()
        for idx, hidUnits in enumerate(n_hidden):
            if idx == 0:
                hidLayer = torch.nn.Linear(n_feature, n_hidden[0])
            else:
                hidLayer = torch.nn.Linear(n_hidden[idx-1], hidUnits)
            self.layers.append(hidLayer)
        self.predict = torch.nn.Linear(n_hidden[-1], n_output)
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.predict(x)
        return x
        
def trainRegressionNet(net, loss_func, optimizer, x, y, nstep = 500):
                                
    plt.ion()   # something about plotting
    
    for t in range(nstep):
        prediction = net(x)     # input x and predict based on x
    
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    
        
        
    
        if t % 25 == 0:
            # plot and show learning process
            print("loss at step" + str(t) + " is:" + str(loss.data))
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
    
    plt.ioff()
    plt.show()    



net1 = NetOneLayer(n_feature=1, n_hidden=20, n_output=1)     # define the network
print(net1)  # net architecture

net2 = NetMultiLayer(n_feature=1, n_hidden=[10, 10, 10], n_output = 1)

print(net2)
     
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss




# torch.manual_seed(1)    # reproducible

x0 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y0 = x0.pow(2) + 0.2*torch.rand(x0.size())      

x1 = x0
y1 = -10*x1.pow(3) + 3*x1.pow(2) + 0.2*torch.rand(x1.size())      

x2 = x0
y2 = -3*x2.pow(4) + 4*x2.pow(3) + 3*x2.pow(2) + 0.2*torch.rand(x2.size())      


           # noisy y data (tensor), shape=(100, 1)
#trainRegressionNet(net1, loss_func, optimizer1, x0, y0)
#trainRegressionNet(net1, loss_func, optimizer1, x1, y1)
#trainRegressionNet(net1, loss_func, optimizer1, x2, y2, nstep = 2000)
trainRegressionNet(net2, loss_func, optimizer2, x2, y2, nstep = 3000)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()



#trainRegressionNet(net2, loss_func, optimizer, x, y)
