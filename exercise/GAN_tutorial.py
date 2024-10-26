#%% GAN Exercise

#%% Library
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#%% HyperParameter
LEARNING_RATE=0.001
EPOCHS=5000

#%% generate sine wave
def generator_sine_wave(seq_length=50, num_samples=1000):
    x=np.linspace(0, 2*np.pi,seq_length)
    sine_wave=np.sin(x)
    data=np.array([sine_wave for _ in range(num_samples)])
    return torch.tensor(data,dtype=torch.float32)

#%% real wave
real_data=generator_sine_wave()

#%% Generator
class Generator(nn.Module):
    def __init__(self,input_dim=10,output_dim=50):
        super(Generator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,output_dim),
            nn.Tanh()
        )
    def forward(self,x):
        return self.model(x)
    
#%% Discriminator
class Discriminator(nn.Module):
    def __init__(self,input_dim=50):
        super(Discriminator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.model(x)
    
#%% model initialization
generator = Generator()
discriminator = Discriminator()

#%% Loss Function and Optimizer
criterion=nn.BCELoss()
optimizer_g=optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_d=optim.Adam(discriminator.parameters(),lr=LEARNING_RATE)

#%% Train
for epoch in range(EPOCHS):
    noise=torch.randn((real_data.size(0),10))
    
    # Train Discriminator 
    optimizer_d.zero_grad()
    
    # real data loss
    real_labels = torch.ones(real_data.size(0),1)
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output,real_labels)
    
    # fake data loss
    fake_data = generator(noise)
    fake_labels = torch.zeros(real_data.size(0),1)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, fake_labels)
    
    # whole Discriminator Loss
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()
    
    # Train Generator
    optimizer_g.zero_grad()
    
    # train fake data to real data
    fake_output = discriminator(fake_data)
    g_loss = criterion(fake_output, real_labels)
    g_loss.backward()
    optimizer_g.step()
    
    if (epoch+1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], d_loss:{d_loss.item()}, g_loss:{g_loss.item()}")
        
with torch.no_grad():
    noise = torch.randn(1,10)
    generated_data = generator(noise).numpy().reshape(-1)

plt.plot(generated_data, label="Generated Sine Wave")
plt.plot(real_data[0].numpy(),label="Real Sine Wave")
plt.legend()
plt.show()
