#import libraries
import torch
from torch import nn
from src.dataset import create_dataloader
from src.model import create_model
from src.train import train
from src.utils import save_model

#set params
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

#Get dataloader
train_loader, test_loader, val_loader, class_names = create_dataloader(batch_size=BATCH_SIZE)

#Get model
model = create_model(num_classes=len(class_names))

#Set up loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Train model
import time
start_time = time.time()

train(model, train_loader, test_loader, loss_fn, optimizer, epochs=NUM_EPOCHS, device=device)
end_time = time.time()
total_time = end_time - start_time

print(f"Total time: {total_time:.3f}")

save_model(model, target_path="models", model_name="naira_classifier.pth")
