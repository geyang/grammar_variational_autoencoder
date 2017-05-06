import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hidden_n=100):
        super(Decoder, self).__init__()
        self.gru_1 = nn.GRU(hidden_size=hidden_n)
        self.gru_2 = nn.GRU(hidden_size=hidden_n)
        self.gru_3 = nn.GRU(hidden_size=hidden_n)

        self.reLU = nn.ReLU()  # reLU non-linear unit for the hidden output
        self.sigmoid = nn.Sigmoid()  # sigmoid non-linear unit for the output

    def forward(self, embedded, hidden_1, hidden_2, hidden_3):
        h, hidden_1 = self.gru_1(embedded, hidden_1)
        h, hidden_2 = self.gru_1(h, hidden_2)
        h, hidden_3 = self.gru_1(h, hidden_3)
        return h, hidden_1, hidden_2, hidden_3


class Encoder(nn.Module):
    def __init__(self, ch_1=2, ch_2=3, ch_3=4, z=15):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=ch_1, out_channels=ch_1, kernel_size=3)
        self.bn_1 = nn.BatchNorm1d(ch_1)
        self.conv_2 = nn.Conv1d(in_channels=ch_2, out_channels=ch_2, kernel_size=3)
        self.bn_2 = nn.BatchNorm1d(ch_2)
        self.conv_3 = nn.Conv1d(in_channels=ch_3, out_channels=ch_3, kernel_size=3)
        self.bn_3 = nn.BatchNorm1d(ch_3)

        self.fc_mu = nn.Linear(ch_3, z)
        self.fc_var = nn.Linear(ch_3, z)

    def forward(self, x):
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        return self.fc_mu(x), self.fc_var(x)


from visdom_helper.visdom_helper import Dashboard


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False
        self.dashboard = Dashboard('Variational-Autoencoder-experiment')

    # question: how is the loss function using the mu and variance?
    def forward(self, x, mu, log_var, recon_x):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
