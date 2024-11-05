# demo.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch import nn

def generate_data(num_samples):
    """
    Generate synthetic 2D data points and label them based on whether
    they are inside a circle centered at (0, 0) with radius 0.5.
    """
    X = np.random.uniform(-1, 1, (num_samples, 2))
    Y = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        x, y = X[i]
        if x**2 + y**2 <= 0.5**2:
            Y[i] = 1.0  # Inside the circle
        else:
            Y[i] = 0.0  # Outside the circle
    return X.astype(np.float32), Y

class CircleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
    
    def forward(self, x):
        return self.model(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y.squeeze())
        acc = ((preds > 0.5) == y.squeeze()).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        return optimizer

def main():
    # Step 1: Generate Data
    num_samples = 5000
    X, Y = generate_data(num_samples)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(test_dataset, batch_size=32)

    # Step 2: Initialize Model
    model = CircleClassifier()

    # Step 3: Train the Model
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, train_loader, val_loader)

    # Step 4: Evaluate the Model
    trainer.validate(model, val_loader)

    # Step 5: Visualize the Decision Boundary
    xx, yy = np.mgrid[-1:1:0.01, -1:1:0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid.astype(np.float32))
    with torch.no_grad():
        probs = model(grid_tensor).reshape(xx.shape)
    probs = probs.numpy()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.6)
    plt.colorbar(contour)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap="RdBu", edgecolors='k', alpha=0.5)
    plt.title('Decision Boundary and Training Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    main()

"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│          val_acc          │    0.9890000224113464     │
│         val_loss          │   0.056591153144836426    │
└───────────────────────────┴───────────────────────────┘
"""
