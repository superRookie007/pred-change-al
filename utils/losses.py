import torch
from torch.nn import functional as F

def consistency_loss(output_logits, target_logits, weights=None, consistency_type='mse'):
    """Computes the consistency loss for mean teacher training.
    
    Args:
    output_logits: the output of the model. Expected to be logits before the softmax of sigmoid layer.
    target_logits: target logits.
    weights: a scaling weight given to each
        instance. Can be either an integer, or a torch array. The array can be of shape (num_examples, 1) or (num_examples). 
    consistency_type (str): can be either 'mse' or 'kl' 

    Note:
    - Returns the mean consistency loss.
    """
    if consistency_type == 'mse':
        loss = softmax_mse_loss(output_logits, target_logits, weights)
    elif consistency_type == 'kl':
        loss = softmax_kl_loss(output_logits, target_logits, weights)
    else:
        raise ValueError("consistency_type can only be either 'mse' or 'kl'.")
    return loss
    

def softmax_mse_loss(output_logits, target_logits, weights=None):
    """Takes softmax on both sides and returns MSE loss.
    
    Args:
    output_logits: the output of the model. Expected to be logits before the softmax of sigmoid layer.
    target_logits: target logits.
    weights: a scaling weight given to each
        instance. Can be either an integer, or a torch array. The array can be of shape (num_examples, 1) or (num_examples). 

    Note:
    - Returns the mean by dividing the number of elements in the output.
    """
    assert output_logits.size() == target_logits.size()
    if weights is None:
        weights = 1.0
    else:
        assert len(weights) == len(output_logits), 'The length of the weights must equal that of the output.'
        if len(weights.size()) == 1:
            weights = weights.view(weights.size()[0], 1)
        elif len(weights.size()) == 2:
            assert weights.size()[1] == 1, 'When weights is a two dimensional array, the second dimension must have size 1.'
        else:
            raise ValueError('weights must be either an integer or array of shape (num_examples, 1) or (num_examples).')
    
    
    output_softmax = F.softmax(output_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    loss = F.mse_loss(output_softmax, target_softmax, reduction='none')
    loss = torch.sum(loss * weights)
    return loss / torch.numel(output_softmax)


def softmax_kl_loss(output_logits, target_logits, weights=None):
    """Computes KL divergence loss.
    
    Args:
        output_logits: the output of the model. Expected to be logits before the softmax of sigmoid layer.
        target_logits: target logits.
        weights: a scaling weight given to each
            instance. Can be either an integer, or a torch array. The array can be of shape (num_examples, 1) or (num_examples). 

    Note:
    - torch.nn.functional.kl_div expects the input to be log-probabilities, while
    the target is probabilities(without log).
    - Returns the batch mean of the kl loss.
    """
    assert output_logits.size() == target_logits.size()
    if weights is None:
        weights = 1.0
    else:
        assert len(weights) == len(output_logits), 'The length of the weights must equal that of the output.'
        if len(weights.size()) == 1:
            weights = weights.view(weights.size()[0], 1)
        elif len(weights.size()) == 2:
            assert weights.size()[1] == 1, 'When weights is a two dimensional array, the second dimension must have size 1.'
        else:
            raise ValueError('weights must be either an integer or array of shape (num_examples, 1) or (num_examples).')
    
    output_log_softmax = F.log_softmax(output_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    loss = F.kl_div(output_log_softmax, target_softmax, reduction='none')
    loss = torch.sum(loss * weights)
    return loss / output_logits.size()[0]


def vae_loss_function(recon_x, x, mu, logvar):
    '''Reconstruction + KL divergence losses summed over all elements and batch.
    
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    c, h, w = x.size(1), x.size(2), x.size(3)
    BCE = F.binary_cross_entropy(recon_x.view(-1, c*h*w), x.view(-1, c*h*w), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def nll_loss_one_hot(input, target, reduction='mean', weights=None):
    '''The negative log likelihood loss.
        Args:
        input: The expected input is the output of log_softmax.
        target: The ground truth in one hot encoding.
        weights (Tensor, optional): a scaling weight given to each
            instance. Can be either an integer, or a torch array. The array can be of shape (num_examples, 1) or (num_examples).
        reduction (string, optional): Specifies the reduction to apply to the output:
            'mean' or 'sum'. 'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default is 'mean'.
    '''
    assert target.size() == input.size(), "target shape: "+str(target.size()) + " vs input shape: " + str(input.size())
    if target.type() == 'torch.LongTensor':
        target = target.float()
    if weights is None:
        weights = 1.0
    else:
        assert len(weights) == len(target), 'The length of the weights must equal that of the targets.'
        if len(weights.size()) == 1:
            weights = weights.view(weights.size()[0], 1)
        elif len(weights.size()) == 2:
            assert weights.size()[1] == 1, 'When weights is a two dimensional array, the second dimension must have size 1.'
        else:
            raise ValueError('weights must be either an integer or array of shape (num_examples, 1) or (num_examples).')

    losses = target * input
    loss_sum = -torch.sum(losses * weights)

    if reduction == 'mean':
        return loss_sum / target.size(0)
    elif reduction == 'sum':
        return loss_sum
    else:
        raise ValueError('reduction must be either "mean" or "sum"!')