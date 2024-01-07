import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

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
        output = outputs[0]
        return torch.tensor(input), torch.tensor(output)


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, n):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(n, int(n/1))
        # Decoder
        self.decoder = nn.Linear(int(n/1), n)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_AE_mapper(dataset,
                    train_raio: float = 0.8,
                    num_epochs: int = 50,
                    batch_size: int = 64,
                    learning_rate: float = 0.001,
                    loss_func: str = "cosine"):
    # Initialize your custom dataset
    random.shuffle(dataset)
    vibe_vector_size = dataset[0]["input"].shape[0]
    train_dataset = MapDataset(dataset[:round(len(dataset) * train_raio)])
    val_dataset = MapDataset(dataset[round(len(dataset) * train_raio):])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    # Model, optimizer, and loss function
    model = Autoencoder(vibe_vector_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss_func == "cosine":
        cos_emb_loss = nn.CosineEmbeddingLoss()
        def cos_loss(a, b):
            return cos_emb_loss(a, b, torch.ones(a.size(0)))
        loss_function = cos_loss
    elif loss_func == "euclidean":
        loss_function = nn.MSELoss()
    else:
        print(f"unknown loss func {loss_func}")
        exit(1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            reconstructions = model(data)
            loss = loss_function(reconstructions, labels)#, torch.ones(data.size(0)))
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
                loss = loss_function(reconstructions, labels)#, torch.ones(data.size(0)))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Print losses
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    return model