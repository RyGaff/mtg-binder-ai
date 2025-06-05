import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (replace with your specific classes)
class_names = ["class1", "class2", "class3"]  # Example: Update this with actual class labels

# Load the trained model
model = models.resnet50(pretrained=False)  # Use pretrained=True if using ImageNet weights
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("card_identifier_model.pth", map_location=device))  # Ensure file exists
model = model.to(device)
model.eval()


# Prediction function
def predict_card(image_path, model, class_names):
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"Image file not found: {image_path}")
    image = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)

    # Return the label
    return class_names[predicted_idx.item()]


# Example usage
image_path = "path_to_test_image/test_card.jpg"

try:
    predicted_label = predict_card(image_path, model, class_names)
    print(f"Predicted Label: {predicted_label}")
except ValueError as e:
    print(e)