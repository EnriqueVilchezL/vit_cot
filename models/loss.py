import torch
from torch import nn


class ThinkingReward(nn.Module):
    def __init__(
        self,
        gamma: float,
        beta: float,
        epsilon: float,
        num_classes: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.original_gamma = gamma
        self.gamma = gamma
        self.beta = beta
        self.device = device
        self.loss = nn.CrossEntropyLoss().to(device)
        self.epsilon = epsilon
        self.thinking_class_index = num_classes
        self.y_thinking = torch.LongTensor([self.thinking_class_index]).to(device)
        self.count = 0
        self.penalty = 1

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = y_hat.shape[0]
        targets = self.y_thinking.repeat(batch_size)
        thinking_wins = self.beta * self.gamma * self.loss(y_hat, targets)
        # todo: maybe think about another strategy to update gamma
        if torch.any(torch.argmax(y_hat, dim=1)[-1] == self.thinking_class_index):
            self.gamma = max(self.gamma - self.epsilon, 0)
            self.count += 1
        else:
            self.count = 0
            self.gamma = self.original_gamma
        #     self.penalty = 1
        # if self.count % 100 == 0:
        #     self.gamma =  self.penalty * self.original_gamma
        #     self.penalty += 1
        # -a * (x - h)**2 
        return thinking_wins
class ThinkingCurriculumReward(nn.Module):
    def __init__(
        self,
        gamma: float,
        beta: float,
        epsilon: float,
        num_classes: int,
        device: torch.device,
        total_steps: int = 1000,  # Total training steps for curriculum
        decay_start: int = 200,   # When to start decaying thinking reward
    ) -> None:
        super().__init__()
        self.original_gamma = gamma
        self.gamma = gamma
        self.beta = beta
        self.device = device
        self.loss = nn.CrossEntropyLoss().to(device)
        self.epsilon = epsilon
        self.thinking_class_index = num_classes
        self.y_thinking = torch.LongTensor([self.thinking_class_index]).to(device)
        
        # Curriculum learning parameters
        self.step_counter = 0
        self.total_steps = total_steps
        self.decay_start = decay_start
        self.thinking_threshold = 0.5  # When to switch from thinking to answering

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = y_hat.shape[0]
        targets = self.y_thinking.repeat(batch_size)

        #if torch.any(torch.argmax(y_hat, dim=1)[-1] == self.thinking_class_index):
        self.step_counter += 1
        #else:
           # self.counter = 0
        
        # Phase 1: Full thinking encouragement
        if self.step_counter < self.decay_start:
            thinking_factor = 1.0
        # Phase 2: Gradual decay of thinking encouragement
        else:
            progress = min(1.0, (self.step_counter - self.decay_start) / 
                          (self.total_steps - self.decay_start))
            thinking_factor = max(0.0, 1.0 - progress)
        
        # Apply curriculum factor to gamma
        curr_gamma = self.original_gamma * thinking_factor
        
        # Calculate thinking reward
        thinking_wins = self.beta * curr_gamma * self.loss(y_hat, targets)
        
        # Monitor thinking behavior to provide analytics
        thinking_rate = torch.mean((torch.argmax(y_hat, dim=1) == self.thinking_class_index).float())
        
        # Print progress occasionally
        if self.step_counter % 100 == 0:
            print(f"Step {self.step_counter}: thinking_factor={thinking_factor:.3f}, thinking_rate={thinking_rate:.3f}")
            
        return thinking_wins

class WeightedClassificationThinkingRewardLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        beta: float,
        epsilon: float,
        num_classes: int,
        device: torch.device,
        classification_weight: float,
        thinking_weight: float = 1,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.loss = nn.CrossEntropyLoss().to(device)
        self.thinking_reward = ThinkingReward(gamma, beta, epsilon, num_classes, device=device)
        self.device = device
        self.classification_weight = classification_weight
        self.thinking_weight = (1 - classification_weight)
        self.count = 0
        self.thinking_class_index = num_classes
        self.original_classification_weight = classification_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if torch.any(torch.argmax(y_hat, dim=1)[-1] == self.thinking_class_index):
            self.count += 1
            self.classification_weight += min(self.gamma + self.classification_weight, 1)
        else:
            self.count = 0
            self.classification_weight = self.original_classification_weight
 
        self.thinking_weight = (1 - self.classification_weight)
        return self.classification_weight * self.loss(y_hat, y) + self.thinking_weight * self.thinking_reward(y_hat, y)
        # return self.thinking_weight * self.thinking_reward(y_hat, y)
    
class ClassificationThinkingRewardLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        beta: float,
        epsilon: float,
        num_classes: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.loss = nn.CrossEntropyLoss().to(device)
        self.thinking_reward = ThinkingReward(gamma, beta, epsilon, num_classes).to(device)
        self.device = device

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(y_hat, y) + self.thinking_reward(y_hat, y)
