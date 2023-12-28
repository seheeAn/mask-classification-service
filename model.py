import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)
    
class custom_resnet34(nn.Module):
    #input size: 224,224
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self,x):
        x = self.resnet(x)
        return x

class custom_resnet50(nn.Module):
    #input size: 224,224
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self,x):
        x = self.resnet(x)
        return x

class vit_model(nn.Module):
    #input size: 
    def __init__(self, num_classes):
        super().__init__()
        self.vitmodel = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.vitmodel(x)
        return x
    
class EfficientNet_b0(nn.Module): # input size 224 224
    def __init__(self, num_classes):
        super(EfficientNet_b0, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b0')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b0', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization
        self.fc = nn.Linear(config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Applying dropout
        x = self.fc(x)

        return x
    
class EfficientNet_b1(nn.Module): # input size 240 240
    def __init__(self, num_classes):
        super(EfficientNet_b1, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b1')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b1', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization
        self.fc = nn.Linear(config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x)  # Applying dropout
        x = self.fc(x)

        return x

class EfficientNet_b2(nn.Module): # input size 260 260
    def __init__(self, num_classes):
        super(EfficientNet_b2, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b2')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b2', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization
        self.fc = nn.Linear(self.eff_net.config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x) # Applying dropout
        x = self.fc(x)

        return x