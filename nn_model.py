import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True)) # orig
        #self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 1536), nn.ReLU(inplace=True)) #new
        #self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 1500), nn.ReLU(inplace=True)) #new
        #self.fc2 = nn.Linear(2000, 512)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(512, 9)
        #self.fc2 = nn.Linear(512, 3)
        #self.fc2 = nn.Linear(512, 6)
        #self.fc2 = nn.Linear(1536, 3) # new
        #self.fc2 = nn.Linear(1536, 9) # new
        #self.fc3 = nn.Linear(1536, 512) # new optional
        #self.fc2 = nn.Sequential(nn.Linear(1500, 500), nn.ReLU(inplace=True)) #new
        self.fc3 = nn.Linear(512, 9)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        #output = self.fc2(output)
        output = self.fc3(output)
        #output = self.fc4(output)

        return output