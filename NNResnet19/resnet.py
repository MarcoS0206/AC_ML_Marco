import torch 
from torch import nn
import torch.optim as optim

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learningRate = 0.01

class BasicBlock(nn.Module):
    def __init__(self,in_features=64,out_features=64,stride=[1,1],down_sample=False):
        # stride : list 
        # the value at corresponding indices are the strides of corresponding layers in a residual block
        
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_features,out_features,3,stride[0],padding=1,bias=False) #weight layer
        self.bn1 = nn.BatchNorm2d(out_features) #weight layer
        
        self.relu = nn.ReLU(True) #relu
        
        self.conv2 = nn.Conv2d(out_features,out_features,3,stride[1],padding=1,bias=False) #weight layer
        self.bn2 = nn.BatchNorm2d(out_features) #weight layer

        self.down_sample = down_sample
        if down_sample:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_features,out_features,1,2,bias=False),
                    nn.BatchNorm2d(out_features)
                )

    def forward(self,x):
        x0=x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:
            x0 = self.downsample(x0)  
        x = x + x0    # F(x)+x
        x= self.relu(x)
        return x

class ResNet(nn.Module):

    def __init__(self,in_channels=3,num_residual_block=[3,4,6,3],num_class=1000,block_type='normal'):
        super(ResNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)

        # if block_type.lower() == 'bottleneck':    
        #     self.resnet,outchannels = self.__bottlenecks(num_residual_block)
        # else:
        self.resnet,outchannels = self.set_layers(num_residual_block)
    
        #extra layer for 19
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=outchannels,out_features=num_class,bias=True)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        #print("Before Last layer: ",x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #print("After Last layer: ",x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
    
    def set_layers(self,num_residual_block):
        layer=[]
        layer += [BasicBlock()]*num_residual_block[0]
        inchannels=64
        for numOFlayers in num_residual_block[1:]:
            stride = [2,1] #updating the stride, the first layer of residual block
            # will have a stride of two and the 2nd layer of the residual block have 
            # a stride of 1
            downsample=True
            outchannels = inchannels*2
            for _ in range(numOFlayers):
                layer.append(BasicBlock(inchannels,outchannels,stride,down_sample=downsample))
                inchannels = outchannels
                downsample = False 
                stride=[1,1]
            
        return nn.Sequential(*layer),outchannels
    
def  resnet18(**kwargs):
    return ResNet(num_residual_block=[2,2,2,2],**kwargs)

model18 = resnet18()
model18.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model18.parameters(), lr=learningRate, momentum=0.9)