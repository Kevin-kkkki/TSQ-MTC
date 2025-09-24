import torch
import torchvision
from torch.utils.data import DataLoader
from create_mh_for_test import MultiInputResNet18
from datasets import CustomDataset, data_transforms
import argparse
import os

def test_model(model, dataloaders, device):
    model.eval()
    model.to(device)

    datasets = ['opt', 'sar']
    total_correct = 0
    total_images = 0

    for data_type in datasets:
        running_corrects = 0
        total_samples = 0

        dataloader = dataloaders[data_type]
        for inputs, labels, _, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs, task=0 if data_type == 'opt' else 1)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        acc = running_corrects.double() / total_samples
        print(f'{data_type.upper()} Dataset - Accuracy: {acc:.4f}, Number of Images: {total_samples}')

        total_correct += running_corrects
        total_images += total_samples

    total_acc = total_correct.double() / total_images
    print(f'Total Accuracy: {total_acc:.4f}, Total Number of Images: {total_images}')

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet18 Testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--data_dir', type=str, default='../dataset/train_ratio_0.5_SAR_False',
                        help='data dir (default: ../dataset/train_ratio_0.5_SAR_False)')
    parser.add_argument('--ckpt_path', type=str, default='best.pth',
                        help='checkpoint path (default: best.pth)')
    parser.add_argument('--exp_name', type=str, default='exp1',
                        help='exp name')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = os.path.join(args.data_dir, 'valid')  
    test_dataset_opt = CustomDataset(root_dir=test_dir, datatype='RGB', transform=data_transforms['valid'])
    test_dataset_sar = CustomDataset(root_dir=test_dir, datatype='SAR', transform=data_transforms['valid'])

    test_loader_opt = DataLoader(test_dataset_opt, batch_size=args.batch_size, shuffle=False)
    test_loader_sar = DataLoader(test_dataset_sar, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'opt': test_loader_opt, 'sar': test_loader_sar}

    resnet_base = torchvision.models.resnet34(pretrained=False)
    num_classes_opt = len(test_dataset_opt.categories)
    num_classes_sar = len(test_dataset_sar.categories)

    model = MultiInputResNet18(resnet_base, num_classes_opt, num_classes_sar, ckpt_path=args.ckpt_path, quant=True)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    test_model(model, dataloaders, device)

if __name__ == "__main__":
    main()
