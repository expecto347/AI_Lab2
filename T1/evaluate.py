import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load checkpoints into models
def load_model(checkpoint_path, num_classes=200):
    model: models.ResNet = models.resnet18(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path)

    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Prepare dataset
def prepare_data(data_path, batch_size=1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        data_path,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return val_loader

# Compare model outputs
def compare_outputs(data_loader, model_1, model_2, topk=5):
    indices_list = []
    for i, (images, target) in tqdm(enumerate(data_loader), total=10000):
        output1: torch.tensor = model_1(images)
        output2: torch.tensor = model_2(images)

        topk_indices_1 = output1.topk(topk, 1, True, True).indices.flatten().tolist()
        topk_indices_2 = output2.topk(topk, 1, True, True).indices.flatten().tolist()
        intersection = set(topk_indices_1) & set(topk_indices_2)
        if not intersection:
            indices_list.append(i)

        if len(indices_list) == 10:
            break
    return indices_list

# Find image paths
def find_image_paths(path, indices_list):
    image_paths = []
    k = 0
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, folder)):
            continue

        for imgpath in os.listdir(os.path.join(path, folder)):
            if k in indices_list:
                image_paths.append(os.path.join(path, folder, imgpath))
            k += 1
    return image_paths

if __name__ == "__main__":
    # Load the two models
    model_1 = load_model('checkpoint1.pth.tar')
    model_2 = load_model('checkpoint2.pth.tar')

    # Prepare data
    val_loader = prepare_data('tiny-imagenet-200/val')

    # Compare the outputs of the two models
    indices_list = compare_outputs(val_loader, model_1, model_2)
    print(indices_list)

    # Find image paths
    image_paths = find_image_paths('tiny-imagenet-200/val', indices_list)
    for path in image_paths:
        print(path)
