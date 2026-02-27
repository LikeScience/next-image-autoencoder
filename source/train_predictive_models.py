import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

#One-hot encoding of discrete integer variable in range [0, num_classes)
def one_hot_encode(values, num_classes):
    v = torch.as_tensor(values, dtype=torch.long)
    if v.ndim == 0:
      one_hot = torch.zeros(num_classes)
      one_hot[values] = 1
    else: 
        num_values = v.shape[0]
        one_hot = torch.zeros(num_values * num_classes)
        indices = torch.arange(num_values) * num_classes + v
        one_hot[indices] = 1
    return one_hot

def list_argmax(values):
    x = max(values)
    return values.index(x)

# Function to process the entire array (each input array of shape [length])
def process_input(input_array, mode,img_width,img_height,scaling_factor=10,n_classes_action=4,n_actions=1):
    if mode == 'in':
    
        world_map = torch.tensor(input_array[:-n_actions]).float()
        world_map /= scaling_factor #scaling; the inputs go from 0 to 10
        action_encoding = one_hot_encode(input_array[-n_actions:], n_classes_action)
        
        return torch.cat([world_map, action_encoding], dim=0)
    elif mode == 'out':
        return torch.tensor(input_array[:-n_actions])/scaling_factor

def runSGD(net, input_train, target_train, input_test, target_test, device, lr=0.001, criterion='mse',
           n_epochs=10, batch_size=32,notrain=False,seed=73,shuffle=False,hide_plot=False):
  """
  Trains autoencoder network with stochastic gradient descent with Adam
  optimizer and loss criterion. Train samples are shuffled, and loss is
  displayed at the end of each opoch for both MSE and BCE. Plots training loss
  at each minibatch (maximum of 500 randomly selected values).

  Args:
    net (torch network)
        ANN object (nn.Module)

    input_train (torch.Tensor)
        vectorized input images from train set

    input_test (torch.Tensor)
        vectorized input images from test set

    criterion (string)
        train loss: 'bce' or 'mse'

    n_epochs (boolean)
        number of full iterations of training data

    batch_size (integer)
        number of element in mini-batches

    verbose (boolean)
        print final loss

  Returns:
    Nothing.
  """
  #set seeds
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  # 4. PyTorch (GPU/CUDA)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed) # for multi-GPU
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  # 2. Move the network to the device
  net.to(device)
  net.train()
  # 3. Move the main tensors to the device (crucial for initial setup)
  input_train = input_train.to(device)
  target_train = target_train.to(device)
  input_test = input_test.to(device)
  target_test = target_test.to(device)
  # Initialize loss function
  if criterion == 'mse':
    loss_fn = nn.MSELoss()
  elif criterion == 'bce':
    loss_fn = nn.BCELoss()
  elif criterion == 'cel':
    loss_fn = nn.CrossEntropyLoss() 
  else:
    print('Please specify either "mse" or "bce" for loss criterion')

    
  # Move the loss function to the device if it has parameters (CrossEntropyLoss does not, 
  # but it's good practice for others like L1Loss which might have reduction='none')
  loss_fn.to(device)

  # Initialize SGD optimizer
  optimizer = optim.Adam(net.parameters(), lr=lr)

  # Placeholder for loss
  track_loss, track_loss_train, track_loss_test = [], [], []

  print('Epoch', '\t', 'Loss train', '\t', 'Loss test')
  for i in range(n_epochs):
    
    if shuffle:
      idx = np.random.permutation(len(input_train)) #shuffled
    else:
      idx = range(len(input_train)) #unshuffled

    batches_input = torch.split(input_train[idx], batch_size)
    batches_target = torch.split(target_train[idx], batch_size)
    batches = zip(batches_input, batches_target)

    # shuffle_idx = np.random.permutation(len(input_train))
    for batch_input, batch_target in batches:
      output_train = net(batch_input)  # Forward pass on the input batch
      loss = loss_fn(output_train, batch_target)  # Compare output with the target
      optimizer.zero_grad()
      if not notrain:
        loss.backward()
      optimizer.step()
      # Keep track of loss at each epoch
      track_loss += [float(loss)]
    loss_epoch = f'{i+1}/{n_epochs}'
    with torch.no_grad():
      output_train = net(input_train)
      loss_train = loss_fn(output_train, target_train)
      loss_epoch += f'\t {loss_train:.4f}'
      track_loss_train += [loss_train.item()]

      output_test = net(input_test)
      loss_test = loss_fn(output_test, target_test)
      loss_epoch += f'\t\t {loss_test:.4f}'
      track_loss_test += [loss_test.item()]

    print(loss_epoch)

  # Plot loss
  step = int(np.ceil(len(track_loss) / 500))
  input_range = np.arange(0, len(track_loss), step)
  if not hide_plot:
    plt.figure()
    plt.plot(input_range, track_loss[::step], 'C0')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.show()
  net.eval()
  return track_loss_train, track_loss_test
