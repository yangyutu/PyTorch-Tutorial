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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
z = torch.zeros((1000, 1))
for i in range(500, 1000):
    z[i,0] = 1
    x[i,0] -= 1

y = 100*torch.sin(10*x) * z + 100*torch.cos(10*x) * (1 - z) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
y = torch.clamp(y, -1, 1)

x = torch.cat([x, z], dim = 1)
# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=2, n_hidden=100, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(1500):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    print(loss)
    # grad clip is import to avoid explosion
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    print("loss at step" + str(t) + " is:" + str(loss.data))
    

    if t % 250 == 0:
        #  plot and show learning process
        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], prediction.data.numpy())
        plt.pause(0.1)

plt.ioff()
plt.show()
