import os
import torch
import data_setup
import engine
import model_builder
import utils
from torchvision import transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':
    writer = SummaryWriter()
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001
    NUM_WORKERS = os.cpu_count()
    train_dir = r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\going_modular\data\animals\train"
    test_dir = r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\going_modular\data\animals\test"

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    train_data_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.TrivialAugmentWide(num_magnitude_bins=31), transforms.ToTensor()])
    test_data_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir, test_dir = test_dir, train_data_transform = train_data_transforms, test_data_transform = test_data_transforms, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    tiny_vgg_model = model_builder.TinyVGG(input_shape = 3, output_shape = len(class_names), hidden_units = 10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = tiny_vgg_model.parameters(), lr = LEARNING_RATE)  

    images, _ = next(iter(train_dataloader))
    writer.add_graph(model=tiny_vgg_model, input_to_model=images.to(device))
    writer.close()
    engine.train(model = tiny_vgg_model, train_dataloader = train_dataloader, test_dataloader = test_dataloader, optimizer = optimizer, loss_fn = loss_fn, epochs = NUM_EPOCHS)

    utils.save_model(model = tiny_vgg_model, target_dir = r"C:\Users\Rizzam\Documents\CODES\Python\deep_learning\pytorch\going_modular\models", model_name = "tiny_vgg_model.pth")