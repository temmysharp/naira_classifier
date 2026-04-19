import torch
from torch import nn
from sklearn.metrics import accuracy_score
torch.manual_seed(42)

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X).cpu()
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y.cpu())
            acc += accuracy_fn(y_true=y.cpu(), 
                                y_pred=torch.softmax(y_pred, dim=1).argmax(dim=1)) 
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_loss": loss.item(),
            "model_acc": acc}
