import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        #Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.encoder = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        #Decoder
        self.deconv1 = nn.Sequential(
             nn.ConvTranspose2d(512,128,kernel_size=(4,4),stride=2,padding=1),
             nn.BatchNorm2d(128),
             nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=(6,6), stride=4, padding=1)

        self.decoder = nn.Sequential(
                        self.deconv1,
                        self.deconv2,
                        self.deconv3
        )

        self.model = nn.Sequential(self.encoder,
                                   self.decoder)

    def forward(self,x):
        depth = self.model(x)
        return depth