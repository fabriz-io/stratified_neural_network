import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import unique


class BaseFeedForwardNet(nn.Module):
    """ Implements the Neural Network Feed Forward Base Class to be used with
        different Loss Functions.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[100, 100, 100], bias=True):
        """ Initializes network architecture.

        Args:
            input_dim (int): Dimensionality of input vector
            output_dim (int): Dimensionality of output vector
        """

        super(BaseFeedForwardNet, self).__init__()

        current_dim = input_dim
        self.layers = []
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hdim, bias=bias))
            self.layers.append(nn.ReLU())
            current_dim = hdim

        self.layers.append(nn.Linear(current_dim, output_dim, bias=bias))

        self.output = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.output(x)


# ______________________________________________________________________________
class StratifiedPartialLikelihoodLoss(torch.nn.Module):
    """ Implements the Stratified Partial Likelihood loss function as a custom
        torch loss function.
    """

    def __init__(self):
        super(StratifiedPartialLikelihoodLoss, self).__init__()

    def partial_likelihood(self, output, event_time, event_indicator):
        """
        Calculates the negative stratified partial likelihood on
        neural network output.

        Args:
            output (array[float]): Output vector of the neural network
            event_time (array[int]): Survival times of observations
            event_indicator(array[bool]): Censoring indicator
        """

        sorted_ind = np.argsort(event_time)
        output = output[sorted_ind]
        event_indicator = event_indicator[sorted_ind]
        output_uncensored = output[event_indicator]

        accumulated_risk = torch.log(
            torch.flip(torch.cumsum(torch.flip(
                torch.exp(output), [0]), dim=0), [0])
        )

        uncensored_accumulated_risk = accumulated_risk[event_indicator]
        return torch.neg(torch.sum(output_uncensored - uncensored_accumulated_risk))

    def forward(self, output, event_time, event_indicator, strata=None):
        """
        Stratifies the partial likelihood across groups.
        
        Args:
            output (array[float]): Output vector of the neural network
            event_time (array[int]): Survival times of observations
            event_indicator(array[bool]): Censoring indicator
            strata (array[int]):
                Elements indicating the group correspondence of
                observations.
        
        Returns:
            Float value in torch format indicating the calculated loss
            across groups.
        """

        if strata is None:
            strata = np.full(len(event_indicator), 1)

        unique_groups = np.unique(strata)

        p_losses = torch.zeros(len(unique_groups))

        for i, strat in enumerate(unique_groups):
            indices_strata = np.where(strata == strat)[0]
            p_losses[i] = self.partial_likelihood(
                output[indices_strata],
                event_time[indices_strata],
                event_indicator[indices_strata],
            )

        return torch.sum(p_losses)


# ______________________________________________________________________________
class StratifiedRankingLoss(torch.nn.Module):
    """ Implements the Stratified Ranking Loss function
        as a custom torch loss function.
    """

    def __init__(self):
        super(StratifiedRankingLoss, self).__init__()
        self.log_of_2 = torch.log(torch.tensor(2.0))

        self.valid_pairs = 0

    def ranking_loss(self, output, event_time, event_indicator):
        """
        This function calculates the ranking loss for survival times.
        The actual calculation is completely vectorized in order to avoid
        any loops. Credits for this vectorization go to an answer on 
        stackoverflow:
        https://stackoverflow.com/questions/61267484/surprisingly-challenging-numpy-vectorization

        Args:
            output (array[float]): Output vector of the neural network
            event_time (array[int]): Survival times of observations
            event_indicator(array[bool]): Censoring indicator
        
        Returns:
            Float in torch format with calculated ranking loss from
            neural network output.
        """

        sorted_ind = np.argsort(event_time)
        event_indicator = event_indicator[sorted_ind]
        output = output[sorted_ind]
        indices_uncensored = np.where(event_indicator)[0]

        mask = indices_uncensored[:, None] < np.arange(len(output))
        v = output[indices_uncensored, None] - output
        vmasked = v[mask]

        ranking_sum = torch.sum(1.0 + F.logsigmoid(vmasked) / self.log_of_2)

        valid_pairs = (
            np.full(indices_uncensored.shape[0], event_indicator.shape[0])
            - indices_uncensored
            - 1
        )

        return torch.neg(ranking_sum), np.sum(valid_pairs)

    def forward(self, output, event_time, event_indicator, strata=None):
        """
        Stratifies the ranking loss across groups.
        
        Args:
            output (array[float]): Output vector of the neural network
            event_time (array[int]): Survival times of observations
            event_indicator(array[bool]): Censoring indicator
            strata (array[int]):
                Elements indicating the group correspondence of
                observations.
        
        Returns:
            Float value in torch format indicating the calculated loss
            across groups.
        """

        if strata is None:
            strata = np.full(len(event_indicator), 1)

        unique_groups = np.unique(strata)
        p_losses = torch.zeros(len(unique_groups))
        valid_pairs = torch.zeros(len(unique_groups))

        for i, strat in enumerate(unique_groups):
            indices_strata = np.where(strata == strat)[0]

            partial_loss = self.ranking_loss(
                output[indices_strata],
                event_time[indices_strata],
                event_indicator[indices_strata],
            )

            p_losses[i] = partial_loss[0]
            valid_pairs[i] = partial_loss[1]

        return torch.sum(p_losses) / torch.sum(valid_pairs)
