import torch
from torch import nn
import torch.nn.functional as F

class LinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_fac, latent_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, int(input_size * hidden_fac))
        self.layer2 = nn.Linear(int(input_size * hidden_fac), latent_size)

    def forward(self, x):
        return F.relu(self.layer2(F.relu(self.layer1(x))))
    
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

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_fac, latent_size,device):
        super().__init__()
        self.device = device
        self.rnn = nn.RNN(input_size,output_size,nonlinearity='relu')
        
    def forward(self, x):
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, hidden = self.rnn(encoded, hidden)
        return hidden
    
    def get_latent(self, x):
        encoded  = self.encoder.forward(x)
        hidden = torch.randn(1,self.latent_size,device = self.device)
        latent, _ = self.rnn(encoded, hidden)
        return encoded, latent

class AutoencoderRNNSeparateAction(nn.Module):
    def __init__(self, map_size, action_encoding_size, output_size, hidden_fac, latent_size,device):
        super().__init__()
        self.device = device
        self.encoder = LinearEncoder(map_size, hidden_fac, latent_size-action_encoding_size)
        self.rnn = nn.RNN(latent_size,latent_size,nonlinearity='relu')
        self.decoder = LinearDecoder(output_size, hidden_fac, latent_size)
        self.map_size = map_size
        self.latent_size = latent_size
        
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