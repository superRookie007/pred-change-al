"""Returns points that the model is most uncertain about.
This implementation selects points whose predictions change a lot during the training.
"""
import warnings
import time
import numpy as np
import torch
import torch.nn.functional as F
from sampling_methods.sampling_def import SamplingMethod
from utils.data import SequentialSampler



class PredictionChange(SamplingMethod):

    def __init__(self, u_indices, dataset, data_name, model=None, device='cuda'):
        """
        Active learning using uncertainty. The smaller the margin the higher the uncertainty.

        Args:
            u_indices: numpy array containing the indices of the unlabelled examples in dataset.
            model: a pytorch model.
            dataset: an object of torch.utils.data.Dataset.
            data_name: the name of the dataset.
            device (optional): the device to use. By default, we want to use gpu for computation.
        """
        self.u_indices = u_indices
        try:
            assert np.sum(u_indices) == np.sum(dataset.u_indices), "Provided u_indices does not match the u_indices attribute in dataset!"
        except AttributeError:
            warnings.warn("Provided dataset does not have the attribute u_indices!")

        self.name = data_name
        self.device = device

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        self.unlabelled_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            sampler=SequentialSampler(self.u_indices),
            **kwargs)
        self.pred_changes = []
        if model:
            self.previous_preds = self._compute_predictions(model)


    def select_batch_(self, N, skip, **kwargs):
        """
        Select the top N examples with the highest prediction change scores.

        Args:
            N: integer. The number of points to sample.
            skip: integer. Skip some prediction changes in the beginning.
        """
        cum_scores = np.sum(self.pred_changes[skip:], axis=0)

        # np.argsort sorts in ascending order by default
        # we want to choose points with higher cum_scores
        idx = np.argsort(-cum_scores)
        return self.u_indices[idx][0:N]


    def select_batch_from_logs(self, N, skip, path, key):
        """
        Select the top N examples with the highest prediction change scores.
        This method loads the prediction changes from a npz log file.

        Args:
            N: integer. The number of points to sample.
            skip: integer. Skip some prediction changes in the beginning.
            path: string. Path to the logging file (npz).
            key: string. The key for the prediction changes in the npz file.
        """
        # load prediction changes from npz log file.
        logs = np.load(path)
        pred_changes = logs[key]
        # compute accumulated prediction changes
        cum_scores = np.sum(pred_changes[skip:], axis=0)
        # np.argsort sorts in ascending order
        # we want to choose points with higher cum_scores
        idx = np.argsort(-cum_scores)
        return self.u_indices[idx][0:N]


    def _compute_predictions(self, model):
        """Compute predictions for data."""
        model.to(self.device)
        model.eval()
        preds = []
        with torch.no_grad():
            for data, _, _ in self.unlabelled_loader:
                data = data.to(self.device)
                temp = model(data) # model outputs
                preds.append(temp.max(1)[1]) #compute predictions
        preds = torch.cat(preds, dim=0)
        return preds.cpu().numpy()
    
    def compute_pred_changes(self, model):
        """
        Computes the prediction change. Does not update the cumulated prediction change scores.
        Prediction change is 1 if prediction has changed from the last model, otherwise it is 0.
        """
        start = time.time()
        model.eval()
        # predictions of the new model
        preds = self._compute_predictions(model)
        assert preds.shape == self.previous_preds.shape
        # initialise the scores to 1. We will deduct 1 for points whose predictions have not changed.
        pred_changes = np.ones(len(preds))
        # find indices of points whose predictions have not changed from last model
        idx = np.equal(self.previous_preds, preds)
        # the scores
        pred_changes[idx] -= 1
        # update self.previous_preds
        self.previous_preds = preds

        end = time.time()
        print('Computing prediction-change scores for {} unlabelled examples took {:.2f} seconds.\n'.format(len(self.u_indices), end-start))
        return pred_changes

    def compute_and_append_pred_changes(self, model):
        """
        Compute the prediction change scores for the model, AND append prediction change scores.
        """
        # Compute the prediction changes, and append the scores.
        self.pred_changes.append(self.compute_pred_changes(model))


    

