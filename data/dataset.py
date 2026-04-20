import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AlzheimerMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the image classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Mappings based on ADNI standards
        # 0: CN (Cognitively Normal)
        # 1: MCI (Mild Cognitive Impairment)
        # 2: AD (Alzheimer's Disease)
        self.class_mapping = {
            "Non Demented": 0,       # CN
            "Very mild Dementia": 1, # MCI
            "Mild Dementia": 2,      # AD
            "Moderate Dementia": 2   # AD (grouping with AD)
        }

        # Load image paths and labels
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping.")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir, batch_size=32, train_split=0.8):
    """
    Creates and returns DataLoaders for train and validation.
    """
    # 224x224 resizing and standard normalization for ResNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = AlzheimerMRIDataset(root_dir=data_dir, transform=None)
    
    # Check dataset size
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No images found in {data_dir}. Check class mappings and folder.")
        
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    import torch
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Apply specific transforms to splits
    train_dataset.dataset.transform = train_transform
    # For validation, we need a separate dataset instance to avoid applying train transforms
    val_dataset_transformed = AlzheimerMRIDataset(root_dir=data_dir, transform=val_transform)
    val_dataset_transformed.image_paths = [dataset.image_paths[i] for i in val_dataset.indices]
    val_dataset_transformed.labels = [dataset.labels[i] for i in val_dataset.indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def get_federated_dataloaders(data_dir, num_clients=3, batch_size=32):
    """
    Splits the dataset into `num_clients` independent datasets for Federated Learning.
    Returns a list of DataLoaders and one global validation DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = AlzheimerMRIDataset(root_dir=data_dir, transform=transform)
    total_size = len(dataset)
    
    # Reserve 20% for global validation
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    import torch
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Split train_dataset among num_clients
    # Basic IID split (equal chunks)
    client_sizes = [train_size // num_clients] * num_clients
    # Add remainder to last client
    client_sizes[-1] += train_size % num_clients
    
    client_datasets = torch.utils.data.random_split(train_dataset, client_sizes)
    client_loaders = [DataLoader(cd, batch_size=batch_size, shuffle=True) for cd in client_datasets]

    return client_loaders, val_loader
