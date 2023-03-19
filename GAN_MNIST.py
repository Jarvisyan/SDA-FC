from torch import nn

class Generator(nn.Module):
    def __init__(self, class_num, z_dim):
        super().__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim + class_num 
        self.output_dim = 1
        
        #FC_block: input_dim -> 1024 -> 128*7*7
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
            )
        #deconv_block: 128*7*7 -> 64*14*14 -> 1*28*28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            #nn.Tanh(),
            nn.Sigmoid(),
            )
        
    def forward(self, X):
        X = self.fc(X).view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        return self.deconv(X)
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        #conv_block: 1*28*28 -> 64*14*14 -> 128*7*7 -> 256*3*3
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            )
        
        #FC_block: 256*3*3 -> 1024 -> 1
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1), 
            )
        
    def forward(self, X):
        X = self.conv(X).view(-1, 256 * 3 * 3)
        return self.fc(X)