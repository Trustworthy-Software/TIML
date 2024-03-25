import torch
import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet18
from layers import LSCLinear
import torch.nn.functional as F

class CILNet(nn.Module):
    def __init__(self, num_classes, res18_weights="imagenet", LSC=False):
        super(CILNet, self).__init__()
        if res18_weights == "imagenet":
            # self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = resnet18(pretrained=True)
        else:
            self.backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        if LSC:
            self.classifier = LSCLinear(512, num_classes)
        else:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, images, AFC_train_out=False):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        if AFC_train_out:
            features.retain_grad()
        return logits, features

    def incremental_classifier(self, num_classes):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, num_classes, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias


class VectorFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=2048, output_token_size=128, kernel_size=7, stride=4):
        super(VectorFeatureExtractor, self).__init__()
        self.output_token_size = output_token_size
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=kernel_size, stride=stride, padding=1), # Reducing the size
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=1), # Further reduction
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernel_size, stride=stride, padding=1), # Continue reducing
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # More layers can be added if necessary
            nn.Conv1d(16, 16, kernel_size=kernel_size, stride=stride, padding=1), # Continuing the reduction
            nn.BatchNorm1d(16),
            nn.ReLU()
            # Continue adding layers until reaching the approximate size for 2048
        )
        # Adjust the layer sizes
        self.fc = nn.Sequential(
            # nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(16*343, 8*output_token_size),
            nn.BatchNorm1d(8*output_token_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1d_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MalscanTIML(nn.Module):
    def __init__(self, vector_input_size=87944, num_classes=4, feature_size=128):
        super(MalscanTIML, self).__init__()
        self.feature_extractor = VectorFeatureExtractor(vector_input_size, output_token_size=feature_size)
        self.fc = nn.Linear(8*feature_size, feature_size)
        self.classifier = nn.Linear(feature_size, num_classes)

    def forward(self, vector_input):
        vector_features = self.feature_extractor(vector_input)

        final_feature_vector = F.relu(self.fc(vector_features))

        out = self.classifier(final_feature_vector)
        return out, vector_features
    
    def incremental_classifier(self, num_classes):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, num_classes, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias


class ImageFeatureExtractor(nn.Module):
    def __init__(self, feat_size=128):
        super(ImageFeatureExtractor, self).__init__()
        # Adjust the number of output channels and possibly remove some layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),  # Output: 8x128x128
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2), # Output: 16x64x64
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=5, stride=4, padding=2), # Output: 1x16x16
            # nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.proj_feat = nn.Linear(256, feat_size)
        # self.proj_feat = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128)
        # )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten to 1x256, then project down
        x = self.proj_feat(x)  # Additional projection layer to get 1x128
        return x

class MalNetTIML(MalscanTIML):
    def __init__(self, num_classes=4, feature_size=128):
        super(MalscanTIML, self).__init__()
        self.feature_extractor = ImageFeatureExtractor(feat_size=feature_size)
        self.fc = nn.Linear(feature_size, feature_size)
        self.classifier = nn.Linear(feature_size, num_classes)

    def forward(self, vector_input):
        vector_features = self.feature_extractor(vector_input)

        final_feature_vector = F.relu(self.fc(vector_features))

        out = self.classifier(final_feature_vector)
        return out, vector_features

if __name__ == "__main__":
    
    num_classes = 10  # Define the number of output classes
    model = CILNet(num_classes)

    # Dummy input data
    images = torch.randn(8, 3, 256, 256)  # 8 images with size 3x256x256

    # Forward pass
    logits, features = model(images)

    import ipdb; ipdb.set_trace()

