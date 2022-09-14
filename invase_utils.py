import torch
import torch.nn as nn
import numpy as np

def show_model_parameter_num(model):  # check if the model parameters equal to keras equivilant
    total_params = sum(p.numel() for p in model.parameters())
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params:', total_params)
    print('total_params_trainable:', total_params_trainable)

class CCE_loss(nn.Module):  # categorical cross entropy loss
    def __init__(self):
        super(CCE_loss, self).__init__()
    def forward(self, y_pred, y_true):
        return (-(y_pred+1e-5).log() * y_true).sum(dim=1).mean()
            
class Actor_loss(nn.Module):
    def __init__(self, lamda=0.1):
        super(Actor_loss, self).__init__()
        self.lamda = lamda

    def forward(self, y_pred, y_final):
        selection = y_final['selection'] 
        critic_out = y_final['critic_out'] 
        baseline_out = y_final['baseline_out'] 
        ground_truth = y_final['ground_truth'] 

        critic_loss = -torch.sum(ground_truth * torch.log(critic_out + 1e-8), dim=1)
        baseline_loss = -torch.sum(ground_truth * torch.log(baseline_out + 1e-8), dim=1)
        Reward = -(critic_loss - baseline_loss)
        custom_actor_loss = Reward * torch.sum(selection * torch.log(y_pred + 1e-8)+ (1 - selection) * torch.log(1 - y_pred + 1e-8), dim=1) - self.lamda * torch.mean(y_pred, dim=1)
        return (-custom_actor_loss).mean()

class Actor(nn.Module):
    def __init__(self, model_parameters):
        super(Actor, self).__init__()
        self.actor_h_dim = model_parameters["actor_h_dim"]
        self.n_layer = model_parameters["selector_n_layer"]
        self.activation = model_parameters["activation"]
        self.dim = model_parameters["dim"]

        if self.activation == 'relu':
            activation = nn.ReLU()
        elif self.activation == 'selu':
            activation = nn.SELU(  )

        self.first_layer = nn.Sequential(nn.Linear(self.dim, self.actor_h_dim), activation)
        mid_seq = nn.Sequential(nn.Linear(self.actor_h_dim, self.actor_h_dim), activation)
        self.mid_layer_list = [mid_seq for _ in range(self.n_layer - 2)]
        self.mid_layers = nn.Sequential(*self.mid_layer_list)
        self.last_layer = nn.Sequential(nn.Linear(self.actor_h_dim, self.dim), nn.Sigmoid())

    def forward(self, x):
        x = self.first_layer(x)
        x = self.mid_layers(x)
        selection_probability = self.last_layer(x)
        return selection_probability

class Critic_or_Baseline(nn.Module):
    def __init__(self, model_parameters):
        super(Critic_or_Baseline, self).__init__()
        self.critic_h_dim = model_parameters["critic_h_dim"]
        self.n_layer = model_parameters["predictor_n_layer"]
        self.activation = model_parameters["activation"]
        self.label_dim = model_parameters["label_dim"]
        self.dim = model_parameters["dim"]

        if self.activation == 'relu':
            activation = nn.ReLU()
        elif self.activation == 'selu':
            activation = nn.SELU( )

        self.first_layer = nn.Sequential(nn.Linear(self.dim, self.critic_h_dim), activation, nn.BatchNorm1d(self.critic_h_dim))
        mid_seq = nn.Sequential(nn.Linear(self.critic_h_dim, self.critic_h_dim), activation, nn.BatchNorm1d(self.critic_h_dim))
        self.mid_layer_list = [mid_seq for _ in range(self.n_layer - 2)]
        self.mid_layers = nn.Sequential(*self.mid_layer_list)
        self.last_layer = nn.Sequential(nn.Linear(self.critic_h_dim, self.label_dim), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.first_layer(x)
        x = self.mid_layers(x)
        y_hat = self.last_layer(x)
        return y_hat


def bernoulli_sampling(prob: torch.Tensor)-> torch.Tensor:
  """ Sampling Bernoulli distribution by given probability.

  Args:
  - prob: P(Y = 1) in Bernoulli distribution.

  Returns:
  - samples: samples from Bernoulli distribution
  """  

  n, d = prob.shape
  if prob.device.type == 'cuda':
    samples = np.random.binomial(1, prob.cpu().detach().numpy(), (n, d))    
    return torch.tensor(samples.astype(np.float32)).float().cuda()
  else:
    samples = np.random.binomial(1, prob.detach().numpy(), (n, d))    
    return torch.tensor(samples.astype(np.float32)).float()