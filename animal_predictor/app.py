import gradio as gr
import torch
import torch.nn as nn
import torchvision
from going_modular.model_builder import TinyVGG
from torchvision import datasets, transforms
from pathlib import Path
import os

train_dir = Path(r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\animal_predictor\data\animals\train")
examples_path = Path(r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\animal_predictor\data\animals\examples")

temp_data = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
class_names = temp_data.classes

device = "cuda" if torch.cuda.is_available() else "cpu"

tiny_vgg_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

tiny_vgg = TinyVGG(input_shape=3, 
                   output_shape=len(class_names), 
                   hidden_units=10)
tiny_vgg.load_state_dict(torch.load(f=Path(r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\animal_predictor\models\tiny_vgg_model.pth"), map_location=torch.device(device)))
tiny_vgg.to(device)

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
efficientnet_b0 = torchvision.models.efficientnet_b0(weights=weights)

efficientnet_b0.classifier[1] = nn.Linear(in_features=1280, out_features=len(class_names))
efficientnet_b0.load_state_dict(torch.load(f=Path(r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\animal_predictor\models\efficientnet_b0.pth"), map_location=torch.device(device)))
efficientnet_b0.to(device)


# Create dictionaries to hold models and their transforms
models = {"TinyVGG": tiny_vgg, "EfficientNet-B0": efficientnet_b0}
model_transforms = {"TinyVGG": tiny_vgg_transform, "EfficientNet-B0": efficientnet_transform}


def predict(image, model_name):
    model = models[model_name]
    transform = model_transforms[model_name]

    model.eval()
    with torch.inference_mode():
        transformed_image = transform(image).unsqueeze(dim=0)
        pred_logits = model(transformed_image.to(device))
    
    pred_probs = torch.softmax(pred_logits, dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    return pred_labels_and_probs


title = "Animal Vision 🐾"
description = "Compare two models classifying images of dogs, cats, horses, elephants and lions."
example_list = [[str(examples_path / example)] for example in os.listdir(examples_path)]
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload an Animal Image"),
        gr.Dropdown(choices=list(models.keys()), value="TinyVGG", label="Select a Model")
    ],
    outputs=gr.Label(num_top_classes=len(class_names), label="Predictions"),
    title=title,
    description=description,
    examples=example_list
)

demo.launch()