import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = ("cuda" if torch.cuda.is_available else "cpu")

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
def TrainStep(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device : torch.device):
    
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X,y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        train_pred = model(X)
        loss = loss_fn(train_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred= train_pred.argmax(dim =1), y_true=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"TRAIN LOSS: {train_loss} | TRAIN ACCURACY: {train_acc}")
    return train_loss, train_acc
def TestStep(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device : torch.device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y_pred= test_pred.argmax(dim=1), y_true=y)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"TEST LOSS: {test_loss} | TEST ACCURACY: {test_acc}")
    return test_loss, test_acc


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs : int = 5):
    results = {"train_loss" : [], "test_loss" : [], "train_acc" : [], "test_acc" : []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = TrainStep(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn = accuracy_fn,
                                           device = device)
        test_loss, test_acc = TestStep(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        accuracy_fn = accuracy_fn,
                                        device = device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    return results