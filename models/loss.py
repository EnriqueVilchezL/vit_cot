import torch
from torch import nn

class ThinkingReward(nn.Module):
    def __init__(self, gamma: float, beta: float, epsilon: float, num_classes : int):
      self.original_gamma = gamma
      self.gamma = gamma
      self.beta = beta
      self.loss = nn.CrossEntropyLoss()
      self.epsilon = epsilon
      self.y_thinking = torch.zeros(num_classes + 1)
      self.y_thinking[-1] = 1
      
    def forward(self, y_hat : torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        thinking_wins = self.beta * self.gamma * self.loss(y_hat, self.y_thinking)
        # todo: maybe think about another strategy to update gamma
        if torch.argmax(y_hat)[-1] == 1:
            self.gamma = min(self.epsilon + self.gamma, 1.0)
        else:
            self.gamma = self.original_gamma
        return thinking_wins

class ClassificationThinkingRewardLoss(nn.Module):
  def __init__(self, gamma : float, beta : float, epsilon : float, num_classes: int):
    self.gamma = gamma
    self.beta = beta
    self.loss = nn.CrossEntropyLoss()
    self.thinking_reward = ThinkingReward(gamma, beta, epsilon, num_classes)
  
  def forward(self, y_hat : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
      y = torch.cat([y, torch.zeros(y.shape[0], 1)], dim=1)

      return self.loss(y_hat, y) + self.thinking_reward(y_hat, y)