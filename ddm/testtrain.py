import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
from copy import deepcopy
from scipy.stats import truncnorm


class NN():
    def __init__(self, input_size, hidden_size, output_size, training_size, no_train_samples=10, no_pred_samples=100, single_head = False, prev_means=None, learning_rate=0.001, type = 'BNN'):
        self.type = type
        if self.type == 'BNN':
            m1, v1, hidden_size = self.create_weights(input_size, hidden_size, output_size, prev_means)

            self.input_size = input_size
            self.out_size = output_size
            self.size = hidden_size
            self.single_head = single_head

            self.W_m, self.b_m = m1[0], m1[1]
            self.W_v, self.b_v = v1[0], v1[1]

            self.W_last_m, self.b_last_m = [], []
            self.W_last_v, self.b_last_v = [], []


            m2, v2 = self.create_prior(input_size, self.size, output_size)

            self.prior_W_m, self.prior_b_m, = m2[0], m2[1]
            self.prior_W_v, self.prior_b_v = v2[0], v2[1]

            self.prior_W_last_m, self.prior_b_last_m = [], []
            self.prior_W_last_v, self.prior_b_last_v = [], []

            self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
            self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
            self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
            self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None



            self.no_layers = len(self.size) - 1
            self.no_train_samples = no_train_samples
            self.no_pred_samples = no_pred_samples
            self.training_size = training_size
            self.learning_rate = learning_rate

            if prev_means is not None:
                self.init_first_head(prev_means)
            else:
                self.create_head()


            m1.append(self.W_last_m)
            m1.append(self.b_last_m)
            v1.append(self.W_last_v)
            v1.append(self.b_last_v)

            r1 = m1 + v1
            self.weights = [item for sublist in r1 for item in sublist]

            self.optimizer = torch.optim.Adam(self.weights, lr=learning_rate)
        
        elif self.type == 'NN':
            self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(input_size, hidden_size, output_size)
            self.no_layers = len(hidden_size) + 1
            self.weights = self.W + self.b + self.W_last + self.b_last
            self.training_size = training_size
            self.optimizer = torch.optim.Adam(self.weights, lr=learning_rate)
            
    
    
    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        self.training_size = N
        if batch_size > N:
            batch_size = N

        costs = []
        for epoch in range(no_epochs):
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            for i in range(total_batch):
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = torch.Tensor(cur_x_train[start_ind:end_ind, :]).to(device = device)
                batch_y = torch.Tensor(cur_y_train[start_ind:end_ind]).to(device = device)

                self.optimizer.zero_grad()
                cost = self.get_loss(batch_x, batch_y, task_idx)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / total_batch

            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "Loss: ","{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        return costs
    
    
    def get_loss(self, x, y, task_idx):
        if self.type == 'BNN':
            return torch.div(self._KL_term(), self.training_size) - self._logpred(x, y, task_idx)
        elif self.type == 'NN':
            return -self._logpred(x, y, task_idx)
    
    def _prediction(self, inputs, task_idx, no_samples = 10):
        if self.type == 'BNN':
            K = no_samples
            size = self.size

            act = torch.unsqueeze(inputs, 0).repeat([K, 1, 1])
            for i in range(self.no_layers-1):
                din = self.size[i]
                dout = self.size[i+1]
                eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
                eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
                weights = torch.add(eps_w * torch.exp(0.5*self.W_v[i]), self.W_m[i])
                biases = torch.add(eps_b * torch.exp(0.5*self.b_v[i]), self.b_m[i])
                pre = torch.add(torch.einsum('mni,mio->mno', act, weights), biases)
                act = torch.nn.functional.relu(pre)

            din = self.size[-2]
            dout = self.size[-1]

            eps_w = torch.normal(torch.zeros((K, din, dout)), torch.ones((K, din, dout))).to(device = device)
            eps_b = torch.normal(torch.zeros((K, 1, dout)), torch.ones((K, 1, dout))).to(device = device)
            Wtask_m = self.W_last_m[task_idx]
            Wtask_v = self.W_last_v[task_idx]
            btask_m = self.b_last_m[task_idx]
            btask_v = self.b_last_v[task_idx]

            weights = torch.add(eps_w * torch.exp(0.5*Wtask_v),Wtask_m)
            biases = torch.add(eps_b * torch.exp(0.5*btask_v), btask_m)
            act = torch.unsqueeze(act, 3)
            weights = torch.unsqueeze(weights, 1)
            pre = torch.add(torch.sum(act * weights, dim = 2), biases)
            return pre
        elif self.type == 'NN':
            act = inputs
            for i in range(self.no_layers-1):
                pre = torch.add(torch.matmul(act, self.W[i]), self.b[i])
                act = torch.nn.functional.relu(pre)
            pre = torch.add(torch.matmul(act, self.W_last[task_idx]), self.b_last[task_idx])
            return pre

    def prediction_prob(self, x_test, task_idx):
        if self.type == 'NNN':
            prob = torch.nn.functional.softmax(self._prediction(x_test, task_idx, self.no_pred_samples), dim=-1)
        else:
            prob = torch.nn.functional.softmax(self._prediction(x_test, task_idx), dim=-1)
        return prob
    
    def _logpred(self, inputs, targets, task_idx):
        if self.type == 'BNN':
            loss = torch.nn.CrossEntropyLoss()
            pred = self._prediction(inputs, task_idx, self.no_train_samples).view(-1,self.out_size)
            targets = targets.repeat([self.no_train_samples, 1]).view(-1)
            log_liks = -loss(pred, targets)
            log_lik = log_liks.mean()
            return log_lik
        elif self.type == 'NN':
            loss = torch.nn.CrossEntropyLoss()
            pred = self._prediction(inputs, task_idx)
            log_lik = - loss(pred, targets)
            return log_lik
    
    def _KL_term(self):
        if self.type == 'BNN':
            kl = 0
            for i in range(self.no_layers-1):
                din = self.size[i]
                dout = self.size[i+1]
                m, v = self.W_m[i], self.W_v[i]
                m0, v0 = self.prior_W_m[i], self.prior_W_v[i]

                const_term = -0.5 * dout * din
                log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
                mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
                kl += const_term + log_std_diff + mu_diff_term

                m, v = self.b_m[i], self.b_v[i]
                m0, v0 = self.prior_b_m[i], self.prior_b_v[i]

                const_term = -0.5 * dout
                log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
                mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
                kl +=  log_std_diff + mu_diff_term + const_term

            no_tasks = len(self.W_last_m)
            din = self.size[-2]
            dout = self.size[-1]

            for i in range(no_tasks):
                m, v = self.W_last_m[i], self.W_last_v[i]
                m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]

                const_term = - 0.5 * dout * din
                log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
                mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
                kl += const_term + log_std_diff + mu_diff_term

                m, v = self.b_last_m[i], self.b_last_v[i]
                m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]

                const_term = -0.5 * dout
                log_std_diff = 0.5 * torch.sum(torch.log(v0) - v)
                mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / v0)
                kl += const_term + log_std_diff + mu_diff_term
            return kl
        
    
    
    def create_weights(self, in_dim, hidden_size, out_dim, prev_means = None):
        if self.type == 'BNN':
            hidden_size = deepcopy(hidden_size)
            hidden_size.append(out_dim)
            hidden_size.insert(0, in_dim)

            no_layers = len(hidden_size) - 1
            W_m = []
            b_m = []
            W_v = []
            b_v = []

            for i in range(no_layers-1):
                din = hidden_size[i]
                dout = hidden_size[i+1]
                if prev_means is not None:
                    W_m_i = prev_means[0][i].detach().data
                    W_m_i.requires_grad = True
                    bi_m_i = prev_means[1][i].detach().data
                    bi_m_i.requires_grad = True
                else:

                    W_m_i= truncated_normal([din, dout], stddev=0.1, variable=True)
                    bi_m_i= truncated_normal([dout], stddev=0.1, variable=True)

                W_v_i = init_tensor(-6.0,  dout = dout, din = din, variable = True)
                bi_v_i = init_tensor(-6.0,  dout = dout, variable = True)


                W_m.append(W_m_i)
                b_m.append(bi_m_i)
                W_v.append(W_v_i)
                b_v.append(bi_v_i)

            return [W_m, b_m], [W_v, b_v], hidden_size
        
        
        elif self.type == 'NN':
            hidden_size = deepcopy(hidden_size)
            hidden_size.append(out_dim)
            hidden_size.insert(0, in_dim)

            no_layers = len(hidden_size) - 1
            W = []
            b = []
            W_last = []
            b_last = []
            for i in range(no_layers-1):
                din = hidden_size[i]
                dout = hidden_size[i+1]


                Wi_m = truncated_normal([din, dout], stddev=0.1, variable = True)
                bi_m = truncated_normal([dout], stddev=0.1, variable = True)


                W.append(Wi_m)
                b.append(bi_m)

            Wi = truncated_normal([hidden_size[-2], out_dim], stddev=0.1, variable = True)
            bi = truncated_normal([out_dim], stddev=0.1, variable = True)
            W_last.append(Wi)
            b_last.append(bi)
            return W, b, W_last, b_last, hidden_size
    
    def save_weights(self):
        ''' Save weights before training on the coreset before getting the test accuracy '''
        if self.type == 'BNN':
            print("Saving weights before core set training")
            self.W_m_copy = [self.W_m[i].clone().detach().data for i in range(len(self.W_m))]
            self.W_v_copy = [self.W_v[i].clone().detach().data for i in range(len(self.W_v))]
            self.b_m_copy = [self.b_m[i].clone().detach().data for i in range(len(self.b_m))]
            self.b_v_copy = [self.b_v[i].clone().detach().data for i in range(len(self.b_v))]

            self.W_last_m_copy = [self.W_last_m[i].clone().detach().data for i in range(len(self.W_last_m))]
            self.W_last_v_copy = [self.W_last_v[i].clone().detach().data for i in range(len(self.W_last_v))]
            self.b_last_m_copy = [self.b_last_m[i].clone().detach().data for i in range(len(self.b_last_m))]
            self.b_last_v_copy = [self.b_last_v[i].clone().detach().data for i in range(len(self.b_last_v))]

            self.prior_W_m_copy = [self.prior_W_m[i].data for i in range(len(self.prior_W_m))]
            self.prior_W_v_copy = [self.prior_W_v[i].data for i in range(len(self.prior_W_v))]
            self.prior_b_m_copy = [self.prior_b_m[i].data for i in range(len(self.prior_b_m))]
            self.prior_b_v_copy = [self.prior_b_v[i].data for i in range(len(self.prior_b_v))]

            self.prior_W_last_m_copy = [self.prior_W_last_m[i].data for i in range(len(self.prior_W_last_m))]
            self.prior_W_last_v_copy = [self.prior_W_last_v[i].data for i in range(len(self.prior_W_last_v))]
            self.prior_b_last_m_copy = [self.prior_b_last_m[i].data for i in range(len(self.prior_b_last_m))]
            self.prior_b_last_v_copy = [self.prior_b_last_v[i].data for i in range(len(self.prior_b_last_v))]

    
    
    def load_weights(self):
        ''' Re-load weights after getting the test accuracy '''
        if self.type == 'BNN':
            print("Reloading previous weights after core set training")
            self.weights = []
            self.W_m = [self.W_m_copy[i].clone().detach().data for i in range(len(self.W_m))]
            self.W_v = [self.W_v_copy[i].clone().detach().data for i in range(len(self.W_v))]
            self.b_m = [self.b_m_copy[i].clone().detach().data for i in range(len(self.b_m))]
            self.b_v = [self.b_v_copy[i].clone().detach().data for i in range(len(self.b_v))]

            for i in range(len(self.W_m)):
                self.W_m[i].requires_grad = True
                self.W_v[i].requires_grad = True
                self.b_m[i].requires_grad = True
                self.b_v[i].requires_grad = True

            self.weights += self.W_m
            self.weights += self.W_v
            self.weights += self.b_m
            self.weights += self.b_v


            self.W_last_m = [self.W_last_m_copy[i].clone().detach().data for i in range(len(self.W_last_m))]
            self.W_last_v = [self.W_last_v_copy[i].clone().detach().data for i in range(len(self.W_last_v))]
            self.b_last_m = [self.b_last_m_copy[i].clone().detach().data for i in range(len(self.b_last_m))]
            self.b_last_v = [self.b_last_v_copy[i].clone().detach().data for i in range(len(self.b_last_v))]

            for i in range(len(self.W_last_m)):
                self.W_last_m[i].requires_grad = True
                self.W_last_v[i].requires_grad = True
                self.b_last_m[i].requires_grad = True
                self.b_last_v[i].requires_grad = True

            self.weights += self.W_last_m
            self.weights += self.W_last_v
            self.weights += self.b_last_m
            self.weights += self.b_last_v

            self.optimizer = torch.optim.Adam(self.weights, lr=self.learning_rate)
            self.prior_W_m = [self.prior_W_m_copy[i].data for i in range(len(self.prior_W_m))]
            self.prior_W_v = [self.prior_W_v_copy[i].data for i in range(len(self.prior_W_v))]
            self.prior_b_m = [self.prior_b_m_copy[i].data for i in range(len(self.prior_b_m))]
            self.prior_b_v = [self.prior_b_v_copy[i].data for i in range(len(self.prior_b_v))]

            self.prior_W_last_m = [self.prior_W_last_m_copy[i].data for i in range(len(self.prior_W_last_m))]
            self.prior_W_last_v = [self.prior_W_last_v_copy[i].data for i in range(len(self.prior_W_last_v))]
            self.prior_b_last_m = [self.prior_b_last_m_copy[i].data for i in range(len(self.prior_b_last_m))]
            self.prior_b_last_v = [self.prior_b_last_v_copy[i].data for i in range(len(self.prior_b_last_v))]

    
    def clean_weights(self):
        if self.type == 'BNN':
            self.W_m_copy, self.W_v_copy, self.b_m_copy, self.b_v_copy = None, None, None, None
            self.W_last_m_copy, self.W_last_v_copy, self.b_last_m_copy, self.b_last_v_copy = None, None, None, None
            self.prior_W_m_copy, self.prior_W_v_copy, self.prior_b_m_copy, self.prior_b_v_copy = None, None, None, None
            self.prior_W_last_m_copy, self.prior_W_last_v_copy, self.prior_b_last_m_copy, self.prior_b_last_v_copy = None, None, None, None

    def get_weights(self):
        if self.type == 'NN':
            weights = [self.weights[:self.no_layers-1], self.weights[self.no_layers-1:2*(self.no_layers-1)], [self.weights[-2]], [self.weights[-1]]]
            return weights
    
    def create_prior(self, in_dim, hidden_size, out_dim, initial_mean = 0, initial_variance = 1):
        if self.type == 'BNN':
            no_layers = len(hidden_size) - 1
            W_m = []
            b_m = []

            W_v = []
            b_v = []

            for i in range(no_layers - 1):
                din = hidden_size[i]
                dout = hidden_size[i + 1]


                W_m_val = initial_mean * torch.zeros([din, dout]).to(device = device)
                bi_m_val = initial_mean * torch.zeros([dout]).to(device = device)


                W_v_val = initial_variance * init_tensor(1,  dout = dout, din = din )
                bi_v_val =  initial_variance * init_tensor(1,  dout = dout)


                W_m.append(W_m_val)
                b_m.append(bi_m_val)
                W_v.append(W_v_val)
                b_v.append(bi_v_val)

            return [W_m, b_m], [W_v, b_v]
    
    
    def update_prior(self):
        if self.type == 'BNN':
            print("updating prior...")
            for i in range(len(self.W_m)):
                self.prior_W_m[i].data.copy_(self.W_m[i].clone().detach().data)
                self.prior_b_m[i].data.copy_(self.b_m[i].clone().detach().data)
                self.prior_W_v[i].data.copy_(torch.exp(self.W_v[i].clone().detach().data))
                self.prior_b_v[i].data.copy_(torch.exp(self.b_v[i].clone().detach().data))

            length = len(self.W_last_m)

            for i in range(length):
                self.prior_W_last_m[i].data.copy_(self.W_last_m[i].clone().detach().data)
                self.prior_b_last_m[i].data.copy_(self.b_last_m[i].clone().detach().data)
                self.prior_W_last_v[i].data.copy_(torch.exp(self.W_last_v[i].clone().detach().data))
                self.prior_b_last_v[i].data.copy_(torch.exp(self.b_last_v[i].clone().detach().data))

    
    def create_head(self):
        ''''Create new head when a new task is detected'''
        if self.type == 'BNN':
            print("creating a new head")
            din = self.size[-2]
            dout = self.size[-1]

            W_m= truncated_normal([din, dout], stddev=0.1, variable=True)
            b_m= truncated_normal([dout], stddev=0.1, variable=True)
            W_v = init_tensor(-6.0,  dout = dout, din = din, variable= True)
            b_v = init_tensor(-6.0,  dout = dout, variable= True)

            self.W_last_m.append(W_m)
            self.W_last_v.append(W_v)
            self.b_last_m.append(b_m)
            self.b_last_v.append(b_v)


            W_m_p = torch.zeros([din, dout]).to(device = device)
            b_m_p = torch.zeros([dout]).to(device = device)
            W_v_p =  init_tensor(1,  dout = dout, din = din)
            b_v_p = init_tensor(1, dout = dout)

            self.prior_W_last_m.append(W_m_p)
            self.prior_W_last_v.append(W_v_p)
            self.prior_b_last_m.append(b_m_p)
            self.prior_b_last_v.append(b_v_p)
            self.weights = []
            self.weights += self.W_m
            self.weights += self.W_v
            self.weights += self.b_m
            self.weights += self.b_v
            self.weights += self.W_last_m
            self.weights += self.W_last_v
            self.weights += self.b_last_m
            self.weights += self.b_last_v
            self.optimizer = torch.optim.Adam(self.weights, lr=self.learning_rate)



    def init_first_head(self, prev_means):
        ''''When the MFVI_NN is instanciated, we initialize weights with those of the Vanilla NN'''
        if self.type == 'BNN':  
            print("initializing first head")
            din = self.size[-2]
            dout = self.size[-1]
            self.prior_W_last_m = [torch.zeros([din, dout]).to(device = device)]
            self.prior_b_last_m = [torch.zeros([dout]).to(device = device)]
            self.prior_W_last_v =  [init_tensor(1,  dout = dout, din = din)]
            self.prior_b_last_v = [init_tensor(1, dout = dout)]

            W_last_m = prev_means[2][0].detach().data
            W_last_m.requires_grad = True
            self.W_last_m = [W_last_m]
            self.W_last_v = [init_tensor(-6.0,  dout = dout, din = din, variable= True)]


            b_last_m = prev_means[3][0].detach().data
            b_last_m.requires_grad = True
            self.b_last_m = [b_last_m]
            self.b_last_v = [init_tensor(-6.0, dout = dout, variable= True)]



def truncated_normal(size, stddev=1, variable = False, mean=0):
    mu, sigma = mean, stddev
    lower, upper= -2 * sigma, 2 * sigma
    X = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
    X_tensor.requires_grad = variable
    return X_tensor

def init_tensor(value,  dout, din = 1, variable = False):
    if din != 1:
        x = value * torch.ones([din, dout]).to(device = device)
    else:
        x = value * torch.ones([dout]).to(device = device)
    x.requires_grad=variable

    return x




def get_scores(model, x_testsets, y_testsets, num_epochs, single_head,  x_coresets, y_coresets, batch_size=None):
    acc = []
    if single_head:
        if len(x_coresets) > 0:
            x_train, y_train = fetch_coreset(x_coresets, y_coresets, single_head)

            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            x_train = torch.Tensor(x_train)
            y_train = torch.Tensor(y_train)

            model.train(x_train, y_train, 0, num_epochs, bsize)

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets)>0:
                model.load_weights()
                x_train, y_train = fetch_coreset(x_coresets[i], y_coresets[i], single_head)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                x_train = torch.Tensor(x_train)
                y_train = torch.Tensor(y_train)
                model.train(x_train, y_train, i, num_epochs, bsize)

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        N = x_test.shape[0]
        bsize = N if (batch_size is None) else batch_size
        cur_acc = 0
        total_batch = int(np.ceil(N * 1.0 / bsize))

        for i in range(total_batch):
            start_ind = i*bsize
            end_ind = np.min([(i+1)*bsize, N])
            batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
            batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)

            pred = model.prediction_prob(batch_x_test, head)

            pred_mean = pred.mean(0)
            pred_y = torch.argmax(pred_mean, dim=1)
            cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

        cur_acc = float(cur_acc)
        cur_acc /= N
        acc.append(cur_acc)
        print("Accuracy is {}".format(cur_acc))
    return acc


def fetch_coreset(x_coresets, y_coresets, single_head):
    if single_head:
        merged_x, merged_y = x_coresets[0], y_coresets[0]
        for i in range(1, len(x_coresets)):
            merged_x = np.vstack((merged_x, x_coresets[i]))
            merged_y = np.hstack((merged_y, y_coresets[i]))
        return merged_x, merged_y
    else:
        x_coresets, y_coresets = np.array(x_coresets), np.array(y_coresets)
        if x_coresets.ndim == 3:
            x_coresets = np.reshape(x_coresets, (x_coresets.shape[0]*x_coresets.shape[1], x_coresets.shape[2]))
        if y_coresets.ndim == 2:
            y_coresets = np.reshape(y_coresets, (y_coresets.shape[0]*y_coresets.shape[1]))
        return (x_coresets), (y_coresets)


def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score