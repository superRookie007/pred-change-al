'''Define all the data-utility function.'''
from __future__ import print_function
import itertools

from PIL import Image

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data.sampler import Sampler

from utils import utils


def get_wanted_classes(images, labels, wanted_labels):
    images_wanted = []
    labels_wanted = []
    for label in wanted_labels:
        if len(labels.shape) == 2:
            indices = np.where(labels[:, 0] == label)
        elif len(labels.shape) == 1:
            indices = np.where(labels == label)
        else:
            raise ValueError("labels can only be shape of (num_examples,) or (num_examples, 1).")
        images_wanted.append(images[indices])
        labels_wanted.append(labels[indices])
    l_images = torch.cat(images_wanted, 0)
    l_labels = torch.cat(labels_wanted, 0)
    return l_images, l_labels


def split_train(images, labels, n_labeled, wanted_labels, seed=0):
    '''Sample labelled training set. Then create indices for 
        - labelled training set
        - unlabelled training set
        - empty set for novel examples
    '''
    wanted_indices = []
    for label in wanted_labels:
        if len(labels.shape) == 2:
            indices = np.where(labels[:, 0] == label)
        elif len(labels.shape) == 1:
            indices = np.where(labels == label)
        else:
            raise ValueError("labels can only be shape of (num_examples,) or (num_examples, 1).")
        wanted_indices.append(indices[0])
    wanted_indices = np.concatenate(wanted_indices)
    unwanted = np.setdiff1d(np.array(range(len(images))), wanted_indices)
    wanted_images, wanted_labels = images[wanted_indices], labels[wanted_indices]
    unwanted_images, unwanted_labels = images[unwanted], labels[unwanted]

    if n_labeled == len(images):
        images = images
        labels = labels
    elif n_labeled > len(wanted_images):
        raise ValueError("Don't have enough labelled examples for wanted classes.")
    else:
        (rest_imgs, rest_labels), (l_imgs, l_labels) = stratified_sampling(wanted_images, wanted_labels, n_labeled, seed)

        images = torch.cat([l_imgs, rest_imgs, unwanted_images], 0)
        labels = torch.cat([l_labels, rest_labels, unwanted_labels], 0)

    labelled_indices = np.array(range(len(l_imgs)))
    unlabelled_indices = np.array(range(len(l_imgs), len(images)))
    novel_indices = np.array([])

    return images, labels, (labelled_indices, unlabelled_indices, novel_indices)


def calculate_weights_knn(X, labeled_indices, alpha=1.0, func='exp', metric='euclidean', algorithm='auto', leaf_size=30):
    if len(X.size()) == 4:
        X = X.view(X.size(0), X.size(1) * X.size(2) * X.size(3))
    elif len(X.size()) == 3:
        X = X.view(X.size(0), X.size(1) * X.size(2))
    neigh = NearestNeighbors(n_neighbors=1, algorithm=algorithm, leaf_size=leaf_size, metric=metric)
    neigh.fit(X[labeled_indices])
    distances, _ = neigh.kneighbors(X, 1, return_distance=True)
    distances = distances / np.max(distances)
    weights = utils.distance_to_weight(distances, alpha=alpha, func=func)
    return torch.from_numpy(weights.astype(np.float32))


def get_encodings(model, data, name=None, device='cuda', dim=784):
    data = data.float().to(device)
    model = model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        data = data.view(-1, dim)
        for i in range(len(data)):
            if name == 'cifar10':
                mean, _ = model.encode(data[i].view(-1, 3, 32, 32))
            else:
                mean, _ = model.encode(data[i:i+1])
            results.append(mean)
    results = torch.cat(results, 0)
    return results.to('cpu') # save it in cpu


def stratified_sampling(data, target, size, seed=0):
    from sklearn.model_selection import StratifiedShuffleSplit

    """
    Create training data and validation from provided data and target using stratified sampling, so that the class
    balance is kept the same.
    Args:
        data: array-like, shape (n_samples, n_features)
        target: array-like, shape (n_samples,)
        size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the subset. If int, represents the absolute number of samples.
        seed: int
    Return:
        Training and validation sets
    """
    if size==1.0 or size==data.shape[0]:
        return(data, target)
    split = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    for train_index, val_index in split.split(data, target):
        train_data, train_label = data[train_index], target[train_index]
        val_data, val_label = data[val_index], target[val_index]
    return (train_data, train_label), (val_data, val_label)


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    # def __iter__(self): #the unlabelled data(primary_batch)!!!!!
    #     primary_iter = iterate_once(self.primary_indices)
    #     secondary_iter = iterate_eternally(self.secondary_indices)
    #     return (
    #         primary_batch + secondary_batch
    #         for (primary_batch, secondary_batch)
    #         in  zip(grouper(primary_iter, self.primary_batch_size),
    #                 grouper(secondary_iter, self.secondary_batch_size))
    #     )
    def __iter__(self): #the labelled ground truth(secondary_batch) comes first!!!!!
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (secondary_batch, primary_batch)
            in  zip(grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image