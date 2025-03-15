
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

class ThinkingViT(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.vit : torch.nn.Module = models.vit_b_16()
    self.logits = None
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    self.logits = self.vit(x)

    # Modified the encoder source code to return the last attention output
    self.last_attention_out = self.vit.encoder.layers[-1].out_attention[:,0]
    return torch.softmax(self.logits, dim=1)

def reshape_logits_with_padding_as_image(logits: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]  # 1000 for ImageNet
    
    # Find a square that can fit all classes
    side_len = int(num_classes**0.5)
    if side_len**2 < num_classes:
        side_len += 1  # Round up
    
    # Pad logits to make it a perfect square
    padding_needed = side_len**2 - num_classes
    if padding_needed > 0:
        padded_logits = torch.cat([logits, torch.zeros(batch_size, padding_needed, device=logits.device)], dim=1)
    else:
        padded_logits = logits
    
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

def get_vit_preprocessing() -> transforms.Compose: 
  return transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

