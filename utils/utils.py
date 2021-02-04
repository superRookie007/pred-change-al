import math
import argparse
from scipy import stats
import torch
from torch.nn import functional as F
import numpy as np
import time


def train(args, model, device, train_loader, optimizer, epoch, global_step):
    '''Train the model.'''
    # switch to train mode
    model.train()
    count = 0
    cum_losses = 0
    cum_correct = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        batch_idx += 1
        start = time.time()
        # target = utils.one_hot_encoding(target, num_classes=args.num_classes)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        model_out = model(data) # model output logits of the student model

        loss = F.cross_entropy(model_out, target)
        assert not (np.isnan(float(loss)) or float(loss) > 1e8), 'loss explosion: {}'.format(float(loss))
        
        loss.backward()
        optimizer.step()
        global_step += 1

        count += len(data)
        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                output_softmax = F.softmax(model_out, dim=1)
                pred = output_softmax.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(target)

                cum_losses += loss.item() * len(data)
                cum_correct += correct

            end = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)],  Loss: {:.6f},  Train_acc: {:.2f}%,  Time: {:.4f}'.format(
                epoch, count, len(train_loader.dataset.l_indices),
                100. * count / len(train_loader.dataset.l_indices), loss.item(), 100. * accuracy, end-start))

    return global_step, cum_losses/count, cum_correct/count


def test(model, device, test_loader):
    '''Measure the performance of the model.'''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            # target = utils.one_hot_encoding(target, num_classes=args.num_classes)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            # test_loss += F.nll_loss_one_hot(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy, test_loss



def compute_outputs(model, device, data_loader):
    """Compute model outputs for data (expected to be unlabelled data).
    NOTE: Outputs are logits (inputs for softmax or log_softmax)."""
    model.eval()
    # start = time.time()
    output = []
    with torch.no_grad():
        for data, _, _ in data_loader:
            data = data.to(device)
            output.append(model(data)) # model output logits of the student model
    output = torch.cat(output, dim=0)
    # end = time.time()
    # print('\nComputing outputs for {} unlabelled examples took {} seconds.\n'.format(len(data_loader.dataset.u_indices), end-start))
    return output


def predict(model, device, data):
    """Note this method is in full batch mode, not mini-batch."""
    model.eval()
    with torch.no_grad():
        data = data.to(device=device)
        output = model(data)
        preds = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        output_softmax = F.softmax(output, dim=1) 
    return preds, output_softmax


def _kl_div(current, previous, symmetrised=False):
    '''Compute kl divergence. If symmetrised is True, compute symmetrised kl divergence.
    Note: both current and previous are logits.'''
    kl_div = F.softmax(previous, dim=1) * (F.log_softmax(previous, dim=1) - F.log_softmax(current, dim=1))
    kl_div = torch.sum(kl_div, dim=1)
    # print('minimum: {}'.format(torch.max(kl_div)))
    # print(kl_div)
    if symmetrised:
        kl_div_reversed = F.softmax(current, dim=1) * (F.log_softmax(current, dim=1) - F.log_softmax(previous, dim=1))
        kl_div_reversed = torch.sum(kl_div_reversed, dim=1)
        assert kl_div.size() == kl_div_reversed.size()
        kl_div = kl_div + kl_div_reversed
    
    kl_div = torch.clamp(kl_div, min=0)
        
    return kl_div


def compute_mse(model, device, data_loader, previous_outputs, pred_change=False):
    """Computes mse between the current model outputs (softmax) and the previous outputs (softmax).

    """
    model.eval()
    start = time.time()

    outputs = compute_outputs(model, device, data_loader)
    outputs, previous_outputs = outputs.to(device), previous_outputs.to(device)
    assert outputs.size() == previous_outputs.size()

    mse = F.mse_loss(F.softmax(outputs, dim=1), F.softmax(previous_outputs, dim=1), reduction='none')
    mse = torch.sum(mse, dim=1)
    
    # if pred_change is True, we only record scores for examples whose predictions have changed
    if pred_change:
        previous_preds = previous_outputs.max(1)[1]
        current_preds = outputs.max(1)[1]
        mse *= 1 - previous_preds.eq(current_preds).float()

    end = time.time()
    print('\nComputing mse for {} unlabelled examples took {:.2f} seconds.\n\n'.format(len(data_loader.dataset.u_indices), end-start))
    return mse.to('cpu'), outputs.to('cpu')


def compute_kl_div(model, device, data_loader, previous_outputs, symmetrised=True, pred_change=False):
    """Computes KL divergence between the current model outputs (softmax output) and the previous outputs (softmax outputs).
    
    Args:
        model: a pytorch model of pytorch.nn.Module class
        device: a string indicating the device to use. e.g., 'cpu' or 'gpu'.
        dataloader: data loader for unlabelled data.
        prevous_outputs: previous model outputs (logits).
        symmetrised: a boolean indicating whether to use symmetrised kl divergence.
        pred_change: if True, only record scores of examples whose predictions have changed.

    Returns the kl divergence for each data point and the new model outputs.
    """
    model.eval()
    start = time.time()

    outputs = compute_outputs(model, device, data_loader)
    outputs, previous_outputs = outputs.to(device), previous_outputs.to(device)
    assert outputs.size() == previous_outputs.size()

    kl_div = _kl_div(outputs, previous_outputs, symmetrised=symmetrised)
    # if pred_change is True, we only record scores for examples whose predictions have changed
    if pred_change:
        previous_preds = previous_outputs.max(1)[1]
        current_preds = outputs.max(1)[1]
        kl_div *= 1 - previous_preds.eq(current_preds).float()

    end = time.time()
    print('\nComputing kl divergence for {} unlabelled examples took {:.2f} seconds.\n\n'.format(len(data_loader.dataset.u_indices), end-start))   
    return kl_div.to('cpu'), outputs.to('cpu')



def str2bool(arg):
    '''Turn strings into booleans. Used for argument parsing.'''
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def averaged_ranks(*args, reverse=True, average=True):
    """
    Compute the averaged rank for each algorithm across different datasets. It assumes the measurement
    for each algorithm and dataset is reliable. 
    NOTE: this method give the smallest number the rank 1. To reverse the ranking, set reverse to True.
    Parameters
    ----------
    measurements1, measurements2, measurements3... : array_like
        Arrays of measurements.  All of the arrays must have the same number
        of elements.  At least 3 sets of measurements must be given. Each measurements 
        is a collection of measurements across different datasets for an algorithm.
    reverse: boolean
        Reverse the ranking so that the highest score will be ranked as 1.
    average:
        Return average rank if set to True, otherwise return the whole ranking
    Returns
        Averaged rank for each algorithm

    """
    measurements = []
    k = len(args)
    if k < 3:
        raise ValueError('Less than 3 levels.  Friedman test not appropriate.')

    n = len(args[0])
    for i in range(0, k):
        if len(args[i]) != n:
            raise ValueError('Unequal N in friedmanchisquare.  Aborting.')
        if type(args[i]) != np.ndarray:
            measurements.append(np.array(args[i]))
        else:
            measurements.append(args[i])

    # Rank data
    data = np.vstack(measurements).T
    if reverse:
        data = data * -1.0
    data = data.astype(float)
    for i in range(len(data)):
        data[i] = stats.rankdata(data[i])

    if average:
        return np.mean(data, axis=0)
    else:
        return data

def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd


def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.
    Needs matplotlib to work.
    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.
    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.
        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]
        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            # get all pairs
            lsums = len(sums)
            allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
            # remove not significant
            notSig = [(i, j) for i, j in allpairs
                      if abs(sums[i] - sums[j]) <= hsd]
            # keep only longest

            def no_longer(ij_tuple, notSig):
                i, j = ij_tuple
                for i1, j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

            return longest

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]


    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2),
              (begin, distanceh - bigtick / 2)],
             linewidth=0.7)
        line([(end, distanceh + bigtick / 2),
              (end, distanceh - bigtick / 2)],
             linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD({:.2f})".format(cd),
             ha="center", va="bottom")

        # no-significance lines
        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start),
                      (rankpos(ssums[r]) + side, start)],
                     linewidth=2.5)
                start += height

        draw_lines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline), (end, cline)],
             linewidth=2.5)
        line([(begin, cline + bigtick / 2),
              (begin, cline - bigtick / 2)],
             linewidth=2.5)
        line([(end, cline + bigtick / 2),
              (end, cline - bigtick / 2)],
             linewidth=2.5)

    if filename:
        print_figure(fig, filename, **kwargs)