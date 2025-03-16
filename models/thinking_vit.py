import torch
import torchvision.models as models
from .common import add_thought_sequence_to_input

from .vit_source import vit_b_16 as local_vit_b_16

class ThinkingViT(torch.nn.Module):
    def __init__(self, num_classes : int) -> None:
        super().__init__()
        self.vit: torch.nn.Module = models.vit_b_16()
        self.thoughts = {}
        self.mlp = torch.nn.Linear(1000, num_classes + 1)

        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.thoughts["logits"] = self.mlp(self.vit(x))
        # Modified the encoder source code to return the last attention output
        self.thoughts["last_attention_out"] = self.vit.encoder.layers[-1].out_attention[:, 0]
        return torch.softmax(self.thoughts["logits"], dim=1)

class ThinkingLocalViT(torch.nn.Module):
    def __init__(self, num_classes : int) -> None:
        super().__init__()
        self.vit: torch.nn.Module = local_vit_b_16()
        self.thoughts = {}
        self.mlp = torch.nn.Linear(1000, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.thoughts["logits"] = self.mlp(self.vit(x))
        # Modified the encoder source code to return the last attention output
        self.thoughts["last_attention_out"] = self.vit.encoder.layers[-1].out_attention[:, 0]
        return torch.softmax(self.thoughts["logits"], dim=1)


class ContinuousThinkingLocalViT(torch.nn.Module):
    def __init__(self, num_classes : int, device: torch.device) -> None:
        super().__init__()
        self.vit: torch.nn.Module = local_vit_b_16()
        self.thoughts = {}
        self.thoughts_list = []
        self.mlp = torch.nn.Linear(1000, num_classes + 1)
        self.needs_to_think = False
        self._forward_fn = self._act_fn
        self.device = device
    
    def _think_fn(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder_input = self.vit.pre_process_encoder_input(x)

        if len(self.thoughts_list) > 0:
            thoughts_tensor = torch.stack(self.thoughts_list).detach()
        else:
            # Handle empty case - maybe use zeros or skip thoughts
            thoughts_tensor = torch.zeros_like(self.encoder_input)

        x = self.vit.encoder(add_thought_sequence_to_input(self.encoder_input, thoughts_tensor, self.device))

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.vit.heads(x)

        self.thoughts["logits"] = self.mlp(x)

        # Modified the encoder source code to return the last attention output
        self.thoughts["last_attention_out"] = self.vit.encoder.layers[-1].out_attention[:, 0]
        self.thoughts_list.append(self.thoughts["last_attention_out"])

        return torch.softmax(self.thoughts["logits"], dim=1)
    
    def _act_fn(self, x: torch.Tensor) -> torch.Tensor:
        self.thoughts["logits"] = self.mlp(self.vit(x))
        # Modified the encoder source code to return the last attention output
        self.thoughts["last_attention_out"] = self.vit.encoder.layers[-1].out_attention[:, 0]
        self.thoughts_list.append(self.thoughts["last_attention_out"])
        return torch.softmax(self.thoughts["logits"], dim=1)
    
    def think(self):
        self.thoughts = {}
        self.thoughts_list = []
        self.needs_to_think = True
        self._forward_fn = self._think_fn
        
    def act(self):
        self.thoughts = {}
        self.thoughts_list = []
        self.needs_to_think = False
        self._forward_fn = self._act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(x)
