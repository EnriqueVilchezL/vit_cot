import torch

def train(
    model : torch.nn.Module,
    train_loader : torch.utils.data.DataLoader,
    val_loader : torch.utils.data.DataLoader,
    loss : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    epochs : int,
    device : torch.device
):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            train_loss += l.item()

        model.eval()
        val_loss = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            l = loss(y_hat, y)

            val_loss += l.item()

        print(f"Epoch {epoch} - Train loss: {train_loss} - Val loss: {val_loss}")

if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    from common import get_device
    from models.thinking_vit import ThinkingViT, get_vit_preprocessing
    from data.data_modules import CRLeavesDataModule

    device = get_device()

    data_module = CRLeavesDataModule(
        root_dir="dataset/",
        batch_size=32,
        test_size=0.5,
        use_index=True,
        indices_dir="indices/",
        sampling=None,
        train_transform=get_vit_preprocessing(),
        test_transform=get_vit_preprocessing()
    )
    data_module.prepare_data()
    data_module.create_data_loaders()

    model = ThinkingViT().to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.test_loader,
        loss=loss,
        optimizer=optimizer,
        epochs=10,
        device=device
    )

