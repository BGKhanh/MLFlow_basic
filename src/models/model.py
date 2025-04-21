import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.BatchNorm2d(in_planes // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AdaptiveWeightCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(AdaptiveWeightCBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

        # Học trọng số cho channel và spatial attention
        self.channel_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Áp dụng attention với trọng số học được
        ca_weight = torch.sigmoid(self.channel_weight)
        sa_weight = torch.sigmoid(self.spatial_weight)

        channel_refined = x * self.ca(x)
        refined = channel_refined * self.sa(channel_refined)

        # Kết hợp với trọng số
        output = x + ca_weight * (channel_refined - x) + sa_weight * (refined - channel_refined)

        return output

class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ResidualCBAMBlock, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        # Attention module với trọng số tự động học
        self.cbam = AdaptiveWeightCBAM(in_planes, ratio)

        # Skip connection scaling factor
        self.skip_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x

        # Convolutional paths
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Attention mechanism
        out = self.cbam(out)

        # Skip connection với trọng số học được
        skip_scale = torch.sigmoid(self.skip_weight)
        out = out + skip_scale * residual

        return self.relu(out)

def initialize_model(model_name, num_classes, transfer_mode="classifier_only"):
    """
    Initialize a pre-trained model with transfer learning options.
    
    Args:
        model_name (str): Name of the base model ('efficientnet', 'mobilenet', 'regnet' or 'densenet')
        num_classes (int): Number of output classes for the target task
        transfer_mode (str): Transfer learning mode
            - "classifier_only": Only replace the classifier layer
            - "RCBAM": Use ResidualCBAMBlock for enhanced features
            
    Returns:
        model: The initialized PyTorch model
    """
    # Chọn mô hình tương ứng
    if model_name == "efficientnet":
        model = models.efficientnet_v2_l(weights='DEFAULT')
    elif model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights='DEFAULT')
    elif model_name == "densenet":
        model = models.densenet201(weights='DEFAULT')
    elif model_name == "regnet":
        model = models.regnet_y_16gf(weights='IMAGENET1K_SWAG_E2E_V1')
    else:
        raise ValueError("Invalid model name. Choose from: efficientnet, mobilenet, densenet")

    # Freeze toàn bộ backbone gốc
    for param in model.parameters():
        param.requires_grad = False

        
    if model_name == "regnet":
        in_features = model.fc.in_features
    else:
      if isinstance(model.classifier, nn.Linear):
          in_features = model.classifier.in_features
      elif isinstance(model.classifier, nn.Module):
          for layer in model.classifier: #Duyệt qua các lớp con của model
              try:
                  in_features = layer.in_features
                  break
              except AttributeError:
                  pass


    if transfer_mode == "classifier_only":
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    elif transfer_mode == "RCBAM":
        model.features = nn.Sequential(
            model.features,  # Giữ nguyên feature extractor gốc
            ResidualCBAMBlock(in_features, ratio=16)  # Module cải tiến
        )

        # Thay thế classifier mới
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )
        # Chỉ huấn luyện các lớp mới thêm vào
        for param in model.features[-1].parameters():
            param.requires_grad = True


    for param in model.classifier.parameters():
        param.requires_grad = True

    return model