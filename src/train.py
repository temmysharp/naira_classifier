import torch
from torch import nn
from sklearn.metrics import accuracy_score

def train(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):
    """Train a model for a specified number of epochs"""
    print("Training has started...")
    for epoch in range(epochs):
        model.train()

        train_loss, train_acc = 0, 0 

        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y)
            acc = accuracy_score(y.cpu(), y_pred.cpu()) 

            train_loss += loss.item()
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        print(f"\nEpoch {epoch + 1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
        test_loss, test_acc = 0,0
        model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)

                y_logits = model(X)
                y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
                loss = loss_fn(y_logits, y)
                acc = accuracy_score(y.cpu(), y_pred.cpu()) 

                test_loss += loss.item()
                test_acc += acc

            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}")

            
            