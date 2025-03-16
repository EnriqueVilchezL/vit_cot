import torch

from torchvision import transforms
import torch.nn.functional as F
import statistics
import random


def reshape_tensor_with_padding_as_image(
    tensor_to_reshape: torch.Tensor,
) -> torch.Tensor:
    batch_size = tensor_to_reshape.shape[0]
    num_classes = tensor_to_reshape.shape[1]  # 1000 for ImageNet

    # Find a square that can fit all classes
    side_len = int(num_classes**0.5)
    if side_len**2 < num_classes:
        side_len += 1  # Round up

    # Pad logits to make it a perfect square
    padding_needed = side_len**2 - num_classes
    if padding_needed > 0:
        padded_logits = torch.cat(
            [
                tensor_to_reshape,
                torch.zeros(
                    batch_size, padding_needed, device=tensor_to_reshape.device
                ),
            ],
            dim=1,
        )
    else:
        padded_logits = tensor_to_reshape

    # Now we can safely reshape to a square
    reshaped = padded_logits.reshape(batch_size, 1, side_len, side_len)
    target_size = 224

    # The rest of your padding code is correct
    pad_h = target_size - side_len
    pad_w = pad_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # This padding usage is correct
    padded = F.pad(reshaped, (pad_left, pad_right, pad_top, pad_bottom))
    return padded.repeat(1, 3, 1, 1)


def add_thought_sequence_to_input(
    input: torch.Tensor, thought_sequence: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if len(thought_sequence) == 0:
        return input
    
    # Ensure same batch size
    assert input.shape[0] == thought_sequence.shape[0], "Batch sizes must match!"
    
    # Determine how many tokens to replace
    num_thought_tokens = thought_sequence.shape[1]
    num_input_tokens = input.shape[1]
    
    # Safety: Don't replace more than input tokens
    if num_thought_tokens > num_input_tokens:
        raise ValueError("Thought sequence longer than input sequence!")
    
    # Clone to avoid in-place ops if necessary
    modified_input = input.clone()
    
    # Replace first N tokens
    modified_input[:, :num_thought_tokens, :] = thought_sequence
    
    return modified_input

def map_value_to_other_range(x, old_min, old_max, new_min, new_max):
    return ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def get_vit_preprocessing() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
