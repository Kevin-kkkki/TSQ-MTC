import torch
import torch.nn as nn
import torchvision
from quant.lsq_plus import *


class ResNet18InputHead(nn.Module):
    def __init__(self, base_model):
        super(ResNet18InputHead, self).__init__()
        self.input_head = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )

    def forward(self, x):
        return self.input_head(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, base_model):
        super(ResNet18Backbone, self).__init__()
        feature_extractor = list(base_model.children())[4:]
        del feature_extractor[-1]
        self.feature_extractor = nn.Sequential(*feature_extractor)

    def forward(self, x):
        return self.feature_extractor(x)


class MultiInputResNet18(nn.Module):
    def __init__(self, base_model, num_classes1, num_classes2, ckpt_path=None, quant=False):
        super(MultiInputResNet18, self).__init__()
        self.input_head1 = ResNet18InputHead(base_model)
        self.input_head2 = ResNet18InputHead(base_model)
        self.shared_backbone = ResNet18Backbone(base_model)
        self.fc1 = nn.Linear(base_model.fc.in_features, num_classes1) 
        self.fc2 = nn.Linear(base_model.fc.in_features, num_classes2)  
        
        if ckpt_path != '':
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])
        
        if quant:
            self.quantize_conv(self.shared_backbone)

    def forward(self, x, task):
        feature = self.input_head1(x) if task == 0 else self.input_head2(x)

        shared_features = self.shared_backbone(feature).squeeze()

        output = self.fc1(shared_features) if task == 0 else self.fc2(shared_features)
        return output
    
    def get_submodule(self, module, submodule_name):
        names = submodule_name.split(".")
        for name in names:
            module = getattr(module, name)
        return module

    def quantize_conv(self,model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                bias = module.bias is not None
                new_conv = Conv2dLSQ(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
                new_conv.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data.clone()
                parent_module_name, attr_name = name.rsplit('.', 1)
                parent_module = self.get_submodule(model, parent_module_name)
                setattr(parent_module, attr_name, new_conv)


if __name__ == '__main__':
    num_classes = 12
    resnet18 = torchvision.models.resnet18(pretrained=True)
    print(resnet18)
    
    model = MultiInputResNet18(resnet18, num_classes)

    print(model)