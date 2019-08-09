import torch
import torch.nn as nn

class Simple_GRU(nn.Module):
    def __init__(self, input_size, output_size, gen_in_dim, hidden_size, num_layers, dropout, batch_size):
        super(Simple_GRU, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.gen_in_dim = gen_in_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.inp1 = nn.Linear(input_size, 128)
        self.inp2 = nn.Linear(128, gen_in_dim)
        self.gru = nn.GRU(input_size=gen_in_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out1 = nn.Linear(hidden_size, 64)
        self.out2 = nn.Linear(64, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, init=False, batch=True):
        batch_size_run = input.shape[0]
        if(not init):
            x = self.inp1(input)
            x = self.inp2(x)
            x = x.view(batch_size_run, 1, self.gen_in_dim)
        else:
            x = input.view(batch_size_run, 1, self.gen_in_dim)

        x, hidden = self.gru(x, hidden)
        x = x.view(batch_size_run, self.hidden_size)
        x = self.leaky_relu(x)
        x = self.leaky_relu(self.out1(x))
        output = self.tanh(self.out2(x))
        return output, hidden

    def initHidden(self, batch=True):
        batch_size_run = self.batch_size if batch else 1
        return torch.zeros(self.num_layers, batch_size_run, self.hidden_size).cuda()

class Critic(nn.Module):
    def __init__(self, input_shape, dropout, batch_size, wgan=False):
        super(Critic, self).__init__()
        self.batch_size = batch_size

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(input_shape[1], 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.out = nn.Linear(64*input_shape[0]/4, 1)
        self.wgan = wgan
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # x = self.sort(input)
        x = input
        x = x.permute(0, 2, 1)
        x = self.max_pool(self.leaky_relu(self.conv1(x)))
        x = self.max_pool(self.leaky_relu(self.conv2(x)))
        x = x.reshape(input.shape[0], -1)
        x = self.out(x)
        if(self.wgan):
            return x
        else:
            return self.sigmoid(x)

    def sort(self, x):
        s, indx = torch.sort(x[:, :, 0])
        for i in range(len(x)):
            x[i] = x[i][indx[i]]
        return x
