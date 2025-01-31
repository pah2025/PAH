import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, alexnet, mobilenet_v2
import timm  # For EfficientNet and other models
from timm import create_model  # For ViT and other models from the "timm" library


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet50(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (512 for ResNet-50)
        self.num_features = resnet.fc.in_features

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.feature_extractor(x)
        # print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, device="cuda"):
        super().__init__()

        # Load pretrained ResNet-50
        resnet = resnet18(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(resnet.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (512 for ResNet-50)
        self.num_features = resnet.fc.in_features
        #print(self.num_features)
        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.feature_extractor(x)
        # #print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]   

class ReducedResNet18(nn.Module):
    def __init__(self, pretrained=False, device="cuda"):
        super().__init__()

        # Load the standard ResNet-18 model
        resnet = resnet18(pretrained=pretrained)
        
        # Modify the convolutional layers to reduce feature maps
        resnet.conv1 = nn.Conv2d(3, 64 // 3, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.bn1 = nn.BatchNorm2d(64 // 3)
        
        # Adjust the number of feature maps in each block
        self.feature_extractor = nn.Sequential(
            nn.Sequential(*self._reduce_layer(resnet.layer1, 64 // 3, 64 // 3)),
            nn.Sequential(*self._reduce_layer(resnet.layer2, 64 // 3, 128 // 3, stride=2)),
            nn.Sequential(*self._reduce_layer(resnet.layer3, 128 // 3, 256 // 3, stride=2)),
            nn.Sequential(*self._reduce_layer(resnet.layer4, 256 // 3, 512 // 3, stride=2)),
        )

        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features
        self.num_features = 512 // 3
        self.device = device
        self.to(device)

    def _reduce_layer(self, layer, in_channels, out_channels, stride=1):
        modified_blocks = []
        for block in layer:
            block.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            block.bn1 = nn.BatchNorm2d(out_channels)
            block.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            block.bn2 = nn.BatchNorm2d(out_channels)
            if block.downsample is not None:
                block.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            modified_blocks.append(block)
            in_channels = out_channels
        return modified_blocks

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_optimizer_list(self):
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]



class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = mobilenet.features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (1280 for MobileNetV2)
        self.num_features = mobilenet.classifier[1].in_features
        #print(self.num_features)

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through MobileNetV2 backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Use the timm library to load EfficientNetB0
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = self.model.classifier.in_features
        #print(self.num_features)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_optimizer_list(self):
        return [{'params': self.model.parameters(), 'lr': 1e-4}]


class AlexNet(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained AlexNet
        alexnet_ = alexnet(pretrained=pretrained)
        # Remove the fully connected layer and retain only the convolutional backbone
        self.feature_extractor = nn.Sequential(
            *(list(alexnet_.children())[:-2])  # Removes FC and avg pooling
        )
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (256 for AlexNet)
        self.num_features = alexnet_.classifier[6].in_features
        #print(self.num_features)

        self.device = device
        self.to(device)

    def forward(self, x):
        # Pass through AlexNet backbone
        x = self.feature_extractor(x)
        #print(x.shape)
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]

class ViT(nn.Module):
    def __init__(self, pretrained=True, device="cuda"):
        super().__init__()

        # Load pretrained Vision Transformer
        vit = create_model('vit_base_patch16_224', pretrained=pretrained)
        #print(vit)
        # Remove the fully connected layer and retain only the transformer backbone
        self.feature_extractor = vit.forward_features
        
        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set the number of output features (768 for ViT base model)
        self.num_features = vit.head.in_features
        #print(self.num_features)

        # Store the device and move the model to the correct device
        self.device = device
        self.to(device)  # Make sure the model is on the correct device

    def forward(self, x):
        # Ensure the input is on the correct device (same as the model)
        x = x.to(self.device)  # Ensure the input is on the same device as the model

        # Pass through ViT backbone
        x = self.feature_extractor(x)
        
        # Global average pooling to get feature vector
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x
    
    def get_optimizer_list(self):
        # Add a lower learning rate for the pretrained parameters
        return [{'params': self.feature_extractor.parameters(), 'lr': 1e-4}]




# Function to initialize the backbone
def get_backbone(name, pretrained=True, device="cuda"):
    if name == "resnet50":
        print("ResNet50", ResNet50(pretrained, device))
        return ResNet50(pretrained, device)
    if name == "resnet18":
        print("ResNet18", ResNet18(pretrained, device))
        return ResNet18(pretrained, device)
    elif name == "mobilenetv2":
        print("MobileNetV2", MobileNetV2(pretrained, device))
        return MobileNetV2(pretrained, device)
    elif name == "efficientnetb0":
        print("EfficientNetB0", EfficientNetB0(pretrained, device))
        return EfficientNetB0(pretrained, device)
    elif name == "vit":
        print("ViT", ViT(pretrained, device))
        return ViT(pretrained, device)
    elif name == "alexnet":
        print("AlexNet", AlexNet(pretrained, device))
        return AlexNet(pretrained, device)
    elif name == "resnet18_reduced":
        print("ReducedResNet18", ReducedResNet18(pretrained, device))
        return ReducedResNet18(pretrained, device)
    else:
        raise ValueError(f"Backbone {name} is not supported.")

if __name__ == "__main__":
    name = "resnet18"
    backbone = get_backbone(name, pretrained=True, device="cuda")
    #print amount of parameters
    print(sum(p.numel() for p in backbone.parameters() if p.requires_grad))

