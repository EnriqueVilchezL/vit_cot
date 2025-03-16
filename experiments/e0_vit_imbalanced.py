import torch
from typing import Callable


import torch
from typing import Callable


def train_model_to_think(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    reshape_thought_fn: Callable[[torch.Tensor], torch.Tensor],
    thought_type: str,
    data_module,
    class_counts: list[int],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    stage: int = 1,
):
    train_loader = data_module.train_loader
    val_loader = data_module.test_loader
    
    for epoch in range(epochs):
        model.train()
        training_thought_lengths = []
        val_thought_lengths = []
        train_loss = 0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            model.think()

            x, y = x.to(device), y.to(device)

            target_thinking_steps = int(
                map_value_to_other_range(
                    class_counts[y[0].item()], min(class_counts), max(class_counts), 0, 10
                )
            )
            
            for i in range(stage):
                model.think()
                y_hat = model(x)

                thinking_class_index = y_hat.shape[1] - 1
                L = loss(y_hat, torch.tensor([thinking_class_index]).to(device))
                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += L.item()
                optimizer.zero_grad()
                print(f"Thinking: {i}/{target_thinking_steps} Loss: {L.item()}")
            
            # Final classification step
            y_hat = model(x)
            L = loss(y_hat, y)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += L.item()
            optimizer.zero_grad()

            # Accuracy calculation
            preds = torch.argmax(y_hat, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            print(f"Thinking: Done Loss: {L.item()} Thinking steps: {target_thinking_steps}")

        scheduler.step()
        train_acc = total_correct / total_samples * 100
        print(
            f"Epoch {epoch} - Train Loss: {train_loss / len(train_loader):.4f} | Train Accuracy: {train_acc:.2f}%"
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
    from models.common import (
        get_vit_preprocessing,
        reshape_tensor_with_padding_as_image,
        map_value_to_other_range,
    )
    from models.loss import WeightedClassificationThinkingRewardLoss

    from data.data_modules import CRLeavesDataModule, ImagesDataModule
    from models.thinking_vit import ThinkingViT, ThinkingLocalViT, ContinuousThinkingLocalViT
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

    model = ContinuousThinkingLocalViT(len(data_module.class_counts), device=device).to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
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
        thought_type="last_attention_out",
        data_module=data_module,
        class_counts=data_module.class_counts,
        scheduler=scheduler,
        stage=2,
    )

    print("Training thought lengths: ", tl)
    print("Validation thought lengths: ", vl)
    ## training 2 learn how to
