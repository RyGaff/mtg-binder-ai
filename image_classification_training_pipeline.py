import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import KFold

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Custom dataset class that uses filenames as labels
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(".jpg")]
        self.labels = []
        for img in self.image_paths:
            label = os.path.splitext(os.path.basename(img))[0]
            if label:
                self.labels.append(label)
            else:
                import warnings
                warnings.warn(f"Image file {img} has no label, skipping.")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
# Data directory
data_dir = "Data/images"

# Create dataset object
full_dataset = CustomImageDataset(data_dir, transform=data_transforms)

# KFold cross-validation setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize model
def initialize_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# Training and evaluation
num_epochs = 10
batch_size = 16
criterion = nn.CrossEntropyLoss()

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    # Prepare data subsets and loaders
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Map unique string labels to integers for training
    unique_labels = list(set(full_dataset.labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Safely map labels for subsets
def safely_map_labels(indices, dataset, label_to_idx, ignore_missing=False):
    import warnings
    mapped_labels = []
    for idx in indices:
        label = dataset.labels[idx]
        if label in label_to_idx:
            mapped_labels.append(label_to_idx[label])
        else:
            if ignore_missing:
                warnings.warn(f"Label '{label}' not found in `label_to_idx`. Skipping sample.")
            else:
                raise KeyError(f"Label '{label}' not found in `label_to_idx`. "
                               "Please check your dataset or label mapping.")
    return mapped_labels


# Example Usage in Main Code
try:
    train_subset.dataset.labels = safely_map_labels(train_idx, full_dataset, label_to_idx, ignore_missing=True)
    val_subset.dataset.labels = safely_map_labels(val_idx, full_dataset, label_to_idx, ignore_missing=True)
except KeyError as e:
    print(f"Error during label mapping: {e}")
    # Handle the error according to your needs

    num_classes = len(unique_labels)

    # Initialize model
    model = initialize_model(num_classes)

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training/Validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            correct_predictions = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                if len(labels) == 0:
                    print("Skipping batch due to lack of valid labels.")
                    continue
                label_indices = torch.tensor([label_to_idx[l] for l in labels]).to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, label_indices)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(preds == label_indices)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = correct_predictions.double() / len(dataloader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")