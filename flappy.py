import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.number_of_actions = 2
        self.gamma = 0.99
        self.epsilon1 = 0.1
        self.epsilon2 = 0.0001
        self.number_of_iterations = 100000
        self.replay_memory = 10000
        self.minibatch_size = 32
        
        self.conv_layer1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_layer3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.conv_layer3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        
        return out

       
def init_weights(w):
    if type(w) == nn.Conv2d or type(w) == nn.Linear:
        torch.nn.init.uniform(w.weight, -0.01, 0.01)
        w.bias.data.fill_(0.01)       
     
     
def convert_img_to_tensor(img):
    img_tensor = img.transpose(2, 0, 1)
    img_tensor = img_tensor.astype(np.float32)
    img_tensor = torch.from_numpy(img_tensor)
    
    return img_tensor    
   
   
def preprocess(img):
    img = img[0:288, 0:404]
    img_data = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY)
    img_data[img_data > 0] = 255
    img_data = np.reshape(img_data, (64, 64, 1))
    
    return img_data   
   
   
def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr = 1e-6)
    
    criterion = nn.MSELoss()
    
    game_state = GameState()
    
    replay_memory = []
    
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    img_data, reward, done = game_state.frame_step(action)
    img_data = preprocess(img_data)
    img_data = convert_img_to_tensor(img_data)
    state = torch.cat((img_data, img_data, img_data, img_data)).unsqueeze(0)
    
    epsilon = model.epsilon1
    iteration = 0
    
    epsilon_decrements = np.linspace(model.epsilon1, 
                                     model.epsilon2, model.number_of_iterations)
    
    while iteration < model.number_of_iterations:
        output = model(state)[0]
        
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        
        random_action = random.random() <= epsilon
        
        if random_action:
            print("Performed random action!")
        
        action_index = [torch.randint(model.number_of_actions, 
                                     torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        
        
        action[action_index] = 1
        
        next_img_data, reward, done = game_state.frame_step(action)
        next_img_data = preprocess(next_img_data)
        next_img_data = convert_img_to_tensor(next_img_data)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], next_img_data)).unsqueeze(0)
        
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        
        
        replay_memory.append((state, action, reward, next_state, terminal))
        
        
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)
                
        epsilon = epsilon_decrements[iteration]
        
        batch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
        
        state_memory = torch.cat(tuple(d[0] for d in batch))
        action_memory = torch.cat(tuple(d[1] for d in batch))
        reward_memory = torch.cat(tuple(d[2] for d in batch))
        next_state_memory = torch.cat(tuple(d[3] for d in batch))
        
        output_memory = model(next_state_memory)
        
        y_memory = torch.cat(tuple(reward_memory[i] if minibatch[i][4] else
                                  reward_memory[i] + model.gamma * torch.max(output_memory[i])
                                  for i in range(len(minibatch))))
        
        q_value = torch.sum(model(state_memory) * action_memory, dim=1)
        
        optimizer.zero_grad()
        
        y_memory = y_memory.detach()
        
        loss = criterion(q_value, y_memory)
        
        loss.backward()
        optimizer.step()
        
        state = next_state
        iteration += 1

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy())) 
        
def test(model):
    game_state = GameState()
    
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    img_data, reward, done = game_state.frame_step(action)
    img_data = preprocess(img_data)
    img_data = convert_img_to_tensor(img_data)
    state = torch.cat((img_data, img_data, img_data, img_data)).unsqueeze(0)
    
    while True:
        
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        
        action_index = torch.argmax(output)
        action[action_index] = 1
        
        next_img_data, reward, done = game_state.frame_step(action)
        next_img_data = preprocess(next_img_data)
        next_img_data = convert_img_to_tensor(img_data)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], next_img_data)).unsqueeze(0)
        
        state = next_state
        
def main(mode):
    cuda_is_available = torch.cuda.is_available()
    
    if mode == 'train':
        model = NeuralNetwork()
        model.apply(init_weights)
        start = time.time()
        
        train(model, start)
        
if __name__ == "__main__":
    main(sys.argv[1])        
