import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from torch.utils.cpp_extension import load

# custom_rnn = load(
#     name="custom_rnn", 
#     sources=["source/custom_rnn.cpp"], 
#     verbose=True
# )

class LinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_fac, latent_size,activation='relu'):
        super().__init__()
        self.layer1 = nn.Linear(input_size, int(input_size * hidden_fac))
        self.layer2 = nn.Linear(int(input_size * hidden_fac), latent_size)
        self.activation=activation

    def forward(self, x):
        if self.activation == 'relu':
            return F.relu(self.layer2(F.relu(self.layer1(x))))
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(self.layer2(F.leaky_relu(self.layer1(x))))
    
class LinearDecoder(nn.Module):
    def __init__(self, output_size, hidden_fac, latent_size):
        super().__init__()
        self.layer1 = nn.Linear(latent_size, int(output_size * hidden_fac))
        self.layer2 = nn.Linear(int(output_size *hidden_fac), output_size)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class AutoencoderWithRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_fac, latent_size,device):
        super().__init__()
        self.device = device
        self.encoder = LinearEncoder(input_size, hidden_fac, latent_size)
        self.rnn = nn.RNN(latent_size,latent_size,nonlinearity='relu')
        self.decoder = LinearDecoder(output_size, hidden_fac, latent_size)
        self.latent_size = latent_size
        
    def forward(self, x):
        encoded  = self.encoder.forward(x)
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, hidden = self.rnn(encoded, hidden)
        out = self.decoder(latent)
        return out
    
    def get_latent(self, x):
        encoded  = self.encoder.forward(x)
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, _ = self.rnn(encoded, hidden)
        return encoded, latent
    
class AutoencoderWithoutRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_fac, latent_size,device):
        super().__init__()
        self.device = device
        self.encoder = LinearEncoder(input_size, hidden_fac, latent_size)
        self.decoder = LinearDecoder(output_size, hidden_fac, latent_size)
        self.latent_size = latent_size
        
    def forward(self, x):
        encoded  = self.encoder.forward(x)
        out = self.decoder(encoded)
        return out
    
    def get_latent(self, x):
        encoded  = self.encoder.forward(x)
        return encoded, None


class PredictiveRNN(nn.Module):
    def __init__(self, img_size, act_enc_size, n_hidden_units,device, scaling_factor = 256, eta = 0.03, tau = 2): 
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.n_hidden_units = n_hidden_units
        self.scaling_factor = scaling_factor
        self.eta = eta
        self.tau = tau
        
        self.hidden = nn.Linear(n_hidden_units,n_hidden_units)
        self.input = nn.Linear(img_size+act_enc_size,n_hidden_units)
        self.output = nn.Linear(n_hidden_units,img_size)
        with torch.no_grad():
            self.input.weight.uniform_(-np.sqrt(1/(img_size+act_enc_size)), np.sqrt(1/(img_size+act_enc_size)))
            self.output.weight.uniform_(-np.sqrt(1/(n_hidden_units)), +np.sqrt(1/(n_hidden_units)))
            self.hidden.weight.uniform_(-np.sqrt(1/(n_hidden_units)), +np.sqrt(1/(n_hidden_units)))
            leakage_term = 1.0 - (1.0 / tau)
            self.hidden.weight.add_(torch.eye(n_hidden_units) * leakage_term)
            self.input.bias.zero_()
            self.hidden.bias.zero_()
            self.output.bias.zero_()

    
    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        h = torch.randn(x.size(0), self.n_hidden_units, device=self.device).mul_(self.eta)
        proj_x = self.input(x)
        noise = torch.randn(x.size(0), x.size(1), self.n_hidden_units, device=self.device).mul_(self.eta)
        decoded = torch.empty(x.size(0), x.size(1), self.img_size, device=self.device)
        for t in range(x.size(1)):
            res = proj_x[:, t, :].add(self.hidden(h))
            m = res.mean(-1, keepdims=True)
            s = res.std(-1, keepdims=True) + (1e-5)
            h = torch.relu((res - (m)) / (s) + (noise[:, t, :]))
            decoded[:, t, :] = torch.sigmoid(self.output(h)) * (self.scaling_factor)
        return decoded


    def get_latent(self, x):
        hidden = torch.randn(1,self.n_hidden_units,device = self.device)
        latent = []
        for x_step in x:
            res = self.input(x_step)+self.hidden(hidden) 
            normed = (res - res.mean(dim=-1, keepdim=True)) / (res.std(dim=-1, keepdim=True) + 1e-5)
            hidden = torch.relu(normed + torch.randn_like(normed) * self.eta)    
            latent.append(hidden)
        return None, torch.stack(latent)
    
    # def forward(self, x):
    #     h = torch.randn(1, self.n_hidden_units, device=self.device) * self.eta
    #     proj_x = self.input(x)
    #     decoded = [None] * x.size(0)
    #     for t in range(x.size(0)):
    #         res = proj_x[t].unsqueeze(0) + self.hidden(h)
    #         m, s = res.mean(-1, True), res.std(-1, True) + 1e-5
    #         h = torch.relu((res - m) / s + torch.randn_like(res) * self.eta)
    #         decoded[t] = torch.sigmoid(self.output(h)).squeeze(0) * self.scaling_factor
    #     return torch.stack(decoded).squeeze(1)
    # def forward(self, x):
    #     h = torch.randn(1, self.n_hidden_units, device=self.device)
    #     return custom_rnn.forward(
    #         x, h, self.input.weight, self.input.bias, 
    #         self.hidden.weight, self.hidden.bias, 
    #         self.output.weight, self.output.bias, 
    #         self.eta, self.scaling_factor
    #     )

class AutoencoderRNNSeparateAction(nn.Module):
    def __init__(self, map_size, action_encoding_size, output_size, hidden_fac, latent_size,device, weight_init=False,encoder_activation='relu'):
        super().__init__()
        
        self.device = device
        self.encoder = LinearEncoder(map_size, hidden_fac, latent_size-action_encoding_size,activation=encoder_activation)
        self.rnn = nn.RNN(latent_size,latent_size,nonlinearity='relu')
        self.decoder = LinearDecoder(output_size, hidden_fac, latent_size)
        self.map_size = map_size
        self.latent_size = latent_size

        if weight_init:
            nn.init.kaiming_normal_(self.encoder.layer1.weight, nonlinearity=encoder_activation)
            nn.init.kaiming_normal_(self.encoder.layer2.weight, nonlinearity=encoder_activation)
        
    def forward(self, x):
        x_map, x_action = x[:,:self.map_size], x[:,self.map_size:]
        encoded_map  = self.encoder.forward(x_map)
        encoded = torch.cat([encoded_map,x_action],axis=1)
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, hidden = self.rnn(encoded, hidden)
        out = self.decoder(latent)
        return out
    
    def get_latent(self, x):
        x_map = x[:,:self.map_size]
        encoded_map  = self.encoder.forward(x_map)
        # hidden = torch.randn(1,self.latent_size,device = self.device)
        # latent, _ = self.rnn(encoded_map, hidden)
        return encoded_map, None



class AutoencoderRNNSeparateActionEncoder(nn.Module):
    def __init__(self, map_size, action_encoding_size, output_size, hidden_fac, img_latent_size, action_latent_size, device):
        super().__init__()
        latent_size = img_latent_size+action_latent_size
        self.device = device
        self.img_encoder = LinearEncoder(map_size, hidden_fac, img_latent_size)
        self.act_encoder = LinearEncoder(action_encoding_size, hidden_fac, action_latent_size)
        self.rnn = nn.RNN(latent_size,latent_size,nonlinearity='relu')
        self.decoder = LinearDecoder(output_size, hidden_fac, latent_size)
        self.map_size = map_size
        self.latent_size = latent_size
        
    def forward(self, x):
        x_map, x_action = x[:,:self.map_size], x[:,self.map_size:]
        encoded_map  = self.img_encoder.forward(x_map)
        encoded_actions  = self.act_encoder.forward(x_action)
        encoded = torch.cat([encoded_map,encoded_actions],axis=1)
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, hidden = self.rnn(encoded, hidden)
        out = self.decoder(latent)
        return out
    
    def get_latent(self, x):
        x_map = x[:,:self.map_size]
        encoded_map  = self.img_encoder.forward(x_map)
        # hidden = torch.randn(1,self.latent_size,device = self.device)
        # latent, _ = self.rnn(encoded_map, hidden)
        return encoded_map, None

class AutoencoderInputs(nn.Module):
    def __init__(self, map_size,device):
        super().__init__()
        self.device = device
        self.map_size = map_size
        
    def forward(self, x):
        x_map = x[:,:self.map_size]
        return x_map
    
    def get_latent(self, x):
        x_map = x[:,:self.map_size]
        return x_map, None