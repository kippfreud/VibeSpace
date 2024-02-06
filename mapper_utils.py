import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import os

class MapDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (Tensor): A tensor containing the data.
            labels (Tensor): A tensor containing the labels.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx]["input"]
        outputs = self.data[idx]["targets"]
        #output = random.choice(outputs)
        output = outputs[0]
        out = output + np.random.normal(0,0.1,output.shape[0])
        return torch.tensor(input, device='cuda'), torch.tensor(out, dtype=torch.float32, device='cuda')


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, n):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder_1 = nn.Linear(n, int(n/4))
        self.encoder_2 = nn.Linear(int(n/4), int(n/16))
        # Decoder
        self.decoder_1 = nn.Linear(int(n/16), int(n/4))
        self.decoder_2 = nn.Linear(int(n/4), n)

    def forward(self, x):
        x = self.encoder_1(x)
        x = F.relu(x)
        x = self.encoder_2(x)
        x = F.relu(x)
        x = self.decoder_1(x)
        x = F.relu(x)
        x = self.decoder_2(x)
        #x = 4.*torch.tanh(x)
        return x

def train_AE_mapper(dataset,
                    train_raio: float = 0.8,
                    num_epochs: int = 50,
                    batch_size: int = 256,
                    learning_rate: float = 0.001,
                    loss_func: str = "cosine",
                    model_path=None,
                    load_map=False):
    # Initialize your custom dataset
    random.shuffle(dataset)
    vibe_vector_size = dataset[0]["input"].shape[0]

    # Model, optimizer, and loss function
    model = Autoencoder(vibe_vector_size).cuda()
    if load_map and model_path is not None:
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model
        else:
            print("Can't load model, retraining")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = MapDataset(dataset[:round(len(dataset) * train_raio)])
    val_dataset = MapDataset(dataset[round(len(dataset) * train_raio):])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    if loss_func == "cosine":
        cos_emb_loss = nn.CosineEmbeddingLoss()
        def cos_loss(a, b):
            return cos_emb_loss(a, b, torch.ones(a.size(0)).cuda())
        loss_function = cos_loss
    elif loss_func == "euclidean":
        loss_function = nn.MSELoss()
    else:
        print(f"unknown loss func {loss_func}")
        exit(1)

    best_val = None
    patience = 100  # Set your patience for early stopping
    epochs_no_improve = 0  # Counter for epochs with no improvement
    best_model_state = None  # To save the best model state
    sd = 0.003
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data, labels in train_loader:
            data = data.to('cuda')
            labels = labels.to('cuda')
            d_orig = data.clone()
            l_orig = labels.clone()
            for i in range(50):
                d = d_orig.clone()
                l = l_orig.clone()

                d = d + torch.empty(d.size(),device="cuda").normal_(mean=0,std=sd)
                l = l + torch.empty(l.size(),device="cuda").normal_(mean=0,std=sd)

                data = torch.cat((data, d),0)
                labels = torch.cat((labels, l),0)

            # print(data.shape)

            optimizer.zero_grad()
            reconstructions = model(data)
            loss = loss_function(reconstructions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                reconstructions = model(data)
                loss = loss_function(reconstructions, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Check for improvement
        if best_val is None or total_val_loss < best_val:
            best_val = total_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save the best model state
            # print(f"New best validation: {best_val} at epoch {epoch}")
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                model.load_state_dict(best_model_state)  # Restore the best model state
                break

        # Print losses
    if model_path is not None:
        torch.save(model.state_dict(), model_path)
    return model
    # best_val = None
    # # Training loop
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_train_loss = 0
    #     for data, labels in train_loader:
    #         optimizer.zero_grad()
    #         reconstructions = model(data)
    #         loss = loss_function(reconstructions, labels)#, torch.ones(data.size(0)))
    #         loss.backward()
    #         optimizer.step()
    #         total_train_loss += loss.item()
    #
    #     avg_train_loss = total_train_loss / len(train_loader)
    #
    #     # Validation loop
    #     model.eval()
    #     total_val_loss = 0
    #     count = 0
    #     with torch.no_grad():
    #         for data, labels in val_loader:
    #             reconstructions = model(data)
    #             loss = loss_function(reconstructions, labels)
    #             total_val_loss += loss.item()
    #         if best_val is None or total_val_loss < best_val:
    #             best_val = total_val_loss
    #             print(f"New best validation: {best_val} at epoch {epoch}")
    #
    #     avg_val_loss = total_val_loss / len(val_loader)
    #
    #     # Print losses
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    # return model