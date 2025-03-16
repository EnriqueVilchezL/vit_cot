import torch
from typing import Callable


def train_model_to_think(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    reshape_thought_fn: Callable[[torch.Tensor], torch.Tensor],
    thought_type: str,
):
    for epoch in range(epochs):
        model.train()
        training_thought_lengths = []
        val_thought_lengths = [] 
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            thinking_class_index = y_hat.shape[1] - 1

            optimizer.zero_grad()
            L = loss(y_hat, y)
            L.backward()
            optimizer.step()

            train_loss += L.item()

            # if thinking:
            counter = 0
            flag = torch.any(torch.argmax(y_hat, dim=1)[-1] == thinking_class_index)
            while flag:
                if counter == 0:
                    print(f"Training: Started thinking... Loss: {L.item()}")
                elif counter % 10 == 0:
                    print(f"Training: Still thinking, length of thoughts {counter} Loss: {L.item()}")
                optimizer.zero_grad()
                x = reshape_thought_fn(model.thoughts[thought_type].detach())
                y_hat = model(x)
                thinking_class_index = y_hat.shape[1] - 1

                L = loss(y_hat, y)
                L.backward()
                optimizer.step()

                train_loss += L.item()
                counter += 1

                flag = torch.any(torch.argmax(y_hat, dim=1)[-1] == thinking_class_index)
                if not flag:
                    print(f"Training: Finished Thinking, thoughts length {counter} Loss: {L.item()}")
            
            training_thought_lengths.append(counter)

        model.eval()
        val_loss = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            L = loss(y_hat, y)

            val_loss += L.item()
            counter = 0

            flag = torch.any(torch.argmax(y_hat, dim=1)[-1] == thinking_class_index)
            while flag:
                if counter == 0:
                    print("Eval: Started thinking...")
                elif counter % 10 == 0:
                    print(f"Eval: Still thinking, length of thoughts {counter} Loss: {L.item()}")
                x = reshape_thought_fn(model.thoughts[thought_type])
                y_hat = model(x)

                L = loss(y_hat, y)
                val_loss += L.item()
                counter += 1
                flag = torch.any(torch.argmax(y_hat, dim=1)[-1] == thinking_class_index)
                if not flag:
                    print(f"Eval: Finished Thinking, thoughts length {counter}")
            

            
            val_thought_lengths.append(counter)

        print(
            f"Epoch {epoch} - Train loss: {train_loss / len(train_loader)} - Val loss: {val_loss / len(val_loader)}"
        )
    return training_thought_lengths, val_thought_lengths


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
    from models.common import get_vit_preprocessing, reshape_tensor_with_padding_as_image
    from models.thinking_vit import ThinkingViT, ThinkingLocalViT
    from data.data_modules import CRLeavesDataModule
    from models.loss import WeightedClassificationThinkingRewardLoss

    device = get_device()

    data_module = CRLeavesDataModule(
        root_dir="dataset/",
        batch_size=1,
        test_size=0.5,
        use_index=False,
        indices_dir="indices/",
        sampling=None,
        train_transform=get_vit_preprocessing(),
        test_transform=get_vit_preprocessing(),
    )
    data_module.prepare_data()
    data_module.create_data_loaders()

    model = ThinkingLocalViT(len(data_module.class_counts)).to(device)
    loss = WeightedClassificationThinkingRewardLoss(
        beta=1, gamma=1, epsilon=0.01, num_classes=len(data_module.class_counts), device=device, classification_weight=0.001
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ## training 1 learn how to think baka
    tl, vl = train_model_to_think(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.test_loader,
        loss=loss,
        optimizer=optimizer,
        epochs=10,
        device=device,
        reshape_thought_fn=reshape_tensor_with_padding_as_image,
        thought_type="last_attention_out"
    )
    
    print("Training thought lengths: ", tl)
    print("Validation thought lengths: ", vl)
    ## training 2 learn how to
