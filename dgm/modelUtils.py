import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import numpy as np
from tqdm import tqdm

class NNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.mu = nn.Linear(input_dim, output_dim).to(device=device)
        with torch.no_grad():
            self._init_weights(input_dim, output_dim)
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.activation(self.mu(x))

    def _init_weights(self, input_size, output_size, constant=1.0):
        scale = constant * np.sqrt(6.0 / (input_size + output_size))
        assert (output_size > 0)
        nn.init.uniform_(self.mu.weight, -scale, scale)
        nn.init.zeros_(self.mu.bias)





class BNNLayer(NNLayer):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__(input_dim, output_dim, activation)
        self.log_sigma = nn.Linear(input_dim, output_dim).to(device=device)
        with torch.no_grad():
            self._init_log_sigma()

        self.w_standard_normal_sampler = torch.distributions.Normal(torch.zeros(self.mu.weight.shape, device=device), torch.ones(self.mu.weight.shape, device=device))
        self.b_standard_normal_sampler = torch.distributions.Normal(torch.zeros(self.mu.bias.shape, device=device), torch.ones(self.mu.bias.shape, device=device))

        self.sampling = True

    def forward(self, x):
        if self.sampling:
            sampled_W = (self.mu.weight + torch.randn_like(self.mu.weight) * torch.exp(self.log_sigma.weight))
            sampled_b = (self.mu.bias + torch.randn_like(self.mu.bias) * torch.exp(self.log_sigma.bias))
            return self.activation(torch.einsum('ij,bj->bi',[sampled_W, x]) + sampled_b)
        else:
            return super().forward(x)

    def _init_log_sigma(self):
        nn.init.constant_(self.log_sigma.weight, -6.0)
        nn.init.constant_(self.log_sigma.bias, -6.0)

    def get_posterior(self):
        return [(self.mu.weight, self.log_sigma.weight), (self.mu.bias, self.log_sigma.bias)]


class Sampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, mu_log_sigma_vec):
        return mu_log_sigma_vec[:, :self.dim] + torch.randn_like(mu_log_sigma_vec[:, :self.dim]) * torch.exp(mu_log_sigma_vec[:, self.dim:])



def evaluator(model, x_test):
    N = len(x_test)
    bound_tot = 0.0
    bound_var = 0.0
    for data_index in tqdm(range(len(x_test))):
        x = x_test[data_index]
        logp_mean, logp_var = estimation(x, model)
        bound_tot += logp_mean 
        bound_var += logp_var 
    bound_tot /= N
    bound_var /= N
    print(f"test_ll={bound_tot}")
    return (bound_tot, np.sqrt(bound_var / N))


def estimation(x, model):
    x = torch.from_numpy(x).to(device=device)
    x = x.view(-1, 28 ** 2)
    x_rep = x.repeat([100, 1]).to(device=device)

    N = x.size()[0]
    Zs_params = model.encoder(x_rep)
    mu_qz, log_sig_qz = Zs_to_mu_sig(Zs_params)
    z = model.decoder.sampler(Zs_params)
    mu_x = model.decoder_common(model.decoder.decoder_layers(z))
    logp = log_bernoulli(x_rep, mu_x)

    log_prior = log_gaussian_prob(z)
    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
    kl_z = logq - log_prior

    bound = torch.reshape(logp - kl_z, (100, N))
    bound -= torch.max(bound, 0)[0]
    lnorm = torch.log(torch.clamp(torch.mean(torch.exp(bound), 0), 1e-9, np.inf))

    test_ll = lnorm + torch.max(bound, 0)[0]
    test_ll_mean = torch.mean(test_ll).item()
    test_ll_var = torch.mean((test_ll - test_ll_mean) ** 2).item()

    return test_ll_mean, test_ll_var


def KL_div_gaussian(mu_p, log_sig_p, mu_q, log_sig_q):
    p_q = torch.exp(-2 * log_sig_q)
    kl = 0.5 * (mu_p - mu_q) ** 2 * p_q - 0.5
    kl += log_sig_q - log_sig_p
    kl += 0.5 * torch.exp(2 * log_sig_p - 2 * log_sig_q)
    return torch.sum(kl, dim=list(range(1, len(kl.shape))))

forced_interval = (1e-9, 1.0)


def log_bernoulli(X, mu_r_x):
    lprob = X * torch.log(torch.clamp(mu_r_x, *forced_interval)) \
              + (1 - X) * torch.log(torch.clamp((1.0 - mu_r_x), *forced_interval))

    return torch.sum(lprob.view(lprob.size()[0], -1), dim=1)  


def log_gaussian_prob(x, mu=torch.zeros(1, device=device), log_sig=torch.zeros(1, device=device)):
    lprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
              - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    return torch.sum(lprob.view(lprob.size()[0], -1), dim=1) 


def Zs_to_mu_sig(Zs_params):
    dimZ = Zs_params.shape[1] // 2  
    mu_qz = Zs_params[:, :dimZ]
    log_sig_qz = Zs_params[:, dimZ:]
    return mu_qz, log_sig_qz

def KL_div_gaussian_from_standard_normal(mu_q, log_sig_q):
    return KL_div_gaussian(mu_q, log_sig_q, torch.zeros(1, device=device), torch.zeros(1, device=device))

def log_P_y_GIVEN_x(Xs, enc, sample_and_decode, NumLogPSamples=100):
    Zs_params = enc(Xs)
    mu_qz, log_sig_qz = Zs_to_mu_sig(Zs_params)
    kl_z = KL_div_gaussian_from_standard_normal(mu_qz, log_sig_qz)
    logp = 0.0
    for _ in range(NumLogPSamples):
        Mu_Ys = sample_and_decode(Zs_params)
        logp += log_bernoulli(Xs, Mu_Ys) / NumLogPSamples
    return logp, kl_z



import math

def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(*shape)
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)
    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def plot_images(images, shape, path, filename, n_rows = 5, color = True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = reshape_and_tile_images(images, shape, n_rows)
    from matplotlib import cm
    plt.imsave(fname=path+filename+"_.png", arr=images, cmap=cm.Greys_r)
    plt.imsave(fname=path+filename+".png", arr=images, cmap='Greys')
    plt.close()


def SampleChange(model, bool):
    for x in model.decoder.decoder_common.decoder_net.children():
        x.sampling = bool
    for x in model.decoder.decoder_layers.children():
        x.sampling = bool

def synthetic_pictures(models):
    with torch.no_grad():
        for task_id, model in enumerate(models):
            SampleChange(model, False)
            pics = model.decoder.sampled_decoder(torch.zeros(25, 100, device=device))
            SampleChange(model, True)
            pics = pics.cpu()
            plot_images(pics, (28, 28), 'dgm/results/figs/', 'iter_'+str(len(models))+'_task_'+str(task_id))
        
        
        row = np.zeros([10, 784])
        for task_id, model in enumerate(models):
            SampleChange(model, False)
            pic = model.decoder.sampled_decoder(torch.zeros(1, 100, device=device))
            SampleChange(model, True)
            pic = pic.cpu()
            row[task_id] = pic
    return row