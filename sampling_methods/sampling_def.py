"""Abstract class for sampling methods.
Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""


import abc
import os
import numpy as np


class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, X, seed, **kwargs):
    self.X = X
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)


  def update_indices(self, dataset, indices, filename="updated_indices"):
      '''Used to update indices for labelled and unlabelled data
      and write the new indices to a file.
      '''
      l_indices = np.concatenate((dataset.l_indices, indices))
      u_indices = np.setdiff1d(dataset.u_indices, indices, assume_unique=True)
      indices_path = os.path.join(dataset.init_folder, filename)
      np.savez_compressed(
          indices_path,
          labelled_indices=l_indices,
          unlabelled_indices=u_indices)