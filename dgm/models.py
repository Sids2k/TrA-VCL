import torch.nn as nn
import torch
from dgm.modelUtils import NNLayer, BNNLayer, Sampler
from tqdm import tqdm
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import itertools
import dgm.modelUtils as modelUtils

class VCL_G_Model(nn.Module):
    def __init__(self, encoder_dimensions, encoder_activations, decoder_dims, decoder_activations, decoder_common):
        super().__init__()
        self.encoder = Encoder(encoder_dimensions, encoder_activations)
        self.decoder = Decoder(decoder_dims, decoder_activations, decoder_common)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.decoder_common = decoder_common
    
    def train_model(self, num_epochs, x_train, y_train, batch_size):
        N = x_train.shape[0]
        self.training_size = N
        if batch_size > N:
            batch_size = N
        self.batch_size = batch_size
        costs = []
        for _ in tqdm(range(num_epochs)):
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]
            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            for i in tqdm(range(total_batch)):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)

                self.optimizer.zero_grad()
                cost = self.get_loss(batch_x)
                cost.backward()
                self.optimizer.step()
                
                avg_cost += cost.item() / total_batch
                
                # TODO commented
                # if i == 2:
                #     break
            
            costs.append(avg_cost)
        self.decoder_common.update_prior()
        return costs
    
    def get_loss(self, x):
        x = x.view(-1, 28*28)
        logp, kl_z = modelUtils.log_P_y_GIVEN_x(x, self.encoder, self.decoder)
        kl_shared_dec_Qt_2_PREV_Qt = self.decoder_common.KL()

        logp_mean = torch.mean(logp)
        kl_z_mean = torch.mean(kl_z)
        kl_Qt_normalized = (kl_shared_dec_Qt_2_PREV_Qt / self.batch_size)
        ELBO = logp_mean - kl_z_mean - kl_Qt_normalized

        return -ELBO
    
class Encoder(nn.Module):
    def __init__(self, encoder_dimensions, encoder_activations):
        super().__init__()
        self.encoder_layers = nn.ModuleList([NNLayer(encoder_dimensions[i], encoder_dimensions[i+1], encoder_activations[i]) for i in range(len(encoder_activations))])
    
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_dims, decoder_activations, decoder_common):
        super().__init__()
        self.decoder_layers = nn.Sequential(*[BNNLayer(decoder_dims[i], decoder_dims[i+1], decoder_activations[i]) for i in range(len(decoder_activations))])
        self.decoder_common = decoder_common
        self.sampler = Sampler(decoder_dims[0])
        self.sampled_decoder = nn.Sequential(*[self.sampler, self.decoder_layers, self.decoder_common])
        
    
    def forward(self, x):
        return self.sampled_decoder(x)

class CommonDecoder(nn.Module):
    def __init__(self, dimensions, activations) -> None:
        super().__init__()
        self.decoder_net = nn.Sequential(*[BNNLayer(dimensions[i], dimensions[i+1], activations[i]) for i in range(len(activations))])
        
        self.prior = [(torch.zeros(mu.shape, device=device), torch.zeros(log_sig.shape, device=device)) for mu, log_sig in list(itertools.chain(*list(map(lambda f: f.get_posterior(), self.decoder_net.children()))))]
        for mu, log_sig in self.prior:
            mu.requires_grad = False
            log_sig.requires_grad = False
    
    def forward(self, x):
        return self.decoder_net(x)
    
    def update_prior(self):
        """
        Copy the current posterior to a constant tensor, which will be used as prior for the next task
        """
        posterior = list(itertools.chain(*list(map(lambda f: f.get_posterior(), self.decoder_net.children()))))
        self.prior = [(mu.clone().detach(), log_sig.clone().detach()) for mu, log_sig in posterior]
        with torch.no_grad():
            for mu_sig in posterior:
                mu_sig[1].fill_(-6.0)
                
    def KL(self):
        params = [(*post, *prior) for (post, prior) in zip(list(itertools.chain(*list(map(lambda f: f.get_posterior(), self.decoder_net.children())))), self.prior)]
        KL_term = torch.tensor(0.0, device=device)
        for param in params:
            unsqueezed_param = list(map(lambda x: x.unsqueeze(0), param))
            tmp = modelUtils.KL_div_gaussian(*unsqueezed_param)
            KL_term += tmp.squeeze()
        return KL_term
    
