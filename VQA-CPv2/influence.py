import torch


def _add_noise_to_tensor(tens: torch.Tensor) -> torch.Tensor:
    """
    Adds noise to a tensor sampled from N(0, tens.std())
    :param tens:
    :return: noisy tensor in the same shape as input
    """
    tens = tens + torch.randn_like(tens) * tens.std(dim=0)  # dim=0 - for each sample in batch

    return tens


def _get_variance(loss: torch.Tensor) -> torch.Tensor:
    """
    Computes the variance along samples for the first dimension in a tensor
    :param loss: [batch, number of evaluate samples]
    :return: variance of a given batch of loss values
    """

    return torch.var(loss, dim=1)


def _get_dif_entropy(loss: torch.Tensor) -> torch.Tensor:
    """
    Computes differential entropy: -E[flogf]
    :param loss:
    :return: a tensor holds the differential entropy for a batch
    """

    return -1 * torch.sum(loss * loss.log())


def _get_entropy(loss: torch.Tensor) -> torch.Tensor:
    """
    Computes functional entropy: E[flogf] - E[f]logE[f]
    :param loss:
    :return: a tensor holds the functional entropy for a batch
    """
    loss = torch.nn.functional.normalize(loss, p=1, dim=1)
    loss = torch.mean(loss * loss.log()) - (torch.mean(loss) * torch.mean(loss).log())

    return loss


def get_batch_statistics(loss: torch.Tensor, evaluate_on_samples: int, reg: str = 'var') -> torch.Tensor:
    """
    Calculate the expectation of the batch gradient
    :param evaluate_on_samples:
    :param loss:
    :param reg:
    :return: Influence expectation
    """
    loss = loss.reshape(-1, evaluate_on_samples)

    if reg == 'var':
        batch_statistics = _get_variance(loss)
        batch_statistics = torch.abs(batch_statistics)
    elif reg == 'ent':
        batch_statistics = _get_entropy(loss)
    elif reg == 'dif_ent':
        batch_statistics = _get_dif_entropy(loss)
    else:
        raise NotImplementedError(f'{reg} is unknown regularization, please use "var" or "ent".')

    batch_statistics = torch.mean(batch_statistics)

    return batch_statistics


def get_batch_norm(grad: torch.Tensor, loss: torch.Tensor = None, norm: int = 1, reg: str = 'var') -> torch.Tensor:
    """
    Calculate the expectation of the batch gradient
    :param loss:
    :param reg:
    :param grad: Tensor holds the gradient batch
    :param norm: The norm we want to use in the expectation term
    :return: Influence expectation
    """
    batch_grad_norm = torch.norm(grad, p=norm, dim=1)
    batch_grad_norm = torch.pow(batch_grad_norm, 2)

    if reg == 'ent':
        batch_grad_norm = batch_grad_norm / loss

    batch_grad_norm = torch.mean(batch_grad_norm)

    return batch_grad_norm


def perturb_tensor(tens: torch.Tensor, num_eval_samples: int, perturbation: bool = True) -> torch.Tensor:
    """
    Flatting the tensor, expanding it, perturbing and reconstructing to the original shape
    :param tens:
    :param num_eval_samples: times to perturb
    :param perturbation: False - only duplicating the tensor
    :return: [batch, samples * num_eval_samples]
    """
    tens_dim = list(tens.shape)

    tens = tens.view(tens.shape[0], -1)
    tens = tens.repeat(1, num_eval_samples)

    tens = tens.view(tens.shape[0] * num_eval_samples, -1)

    if perturbation:
        tens = _add_noise_to_tensor(tens)

    tens_dim[0] *= num_eval_samples

    tens = tens.view(*tens_dim)
    tens.requires_grad_()

    return tens


def _get_normalized_score(inf_scores: torch.Tensor, norm: int, inf_dist: str) -> torch.Tensor:
    """
    Calculate the normalized score with norm in any order or kl divergence
    :param inf_scores: batch of information scores
    :param norm: which order of norm to use. Only norm 1 and 2 have been checked.
    :param inf_dist: possible distance measures 'kl' and 'norm'
    :return:
    """
    inf_scores = torch.nn.functional.normalize(inf_scores.unsqueeze(0), p=1).squeeze(0)

    uniform_scores = (1 / inf_scores.shape[0])
    uniform_scores = uniform_scores * torch.ones_like(inf_scores, requires_grad=True).cuda()

    if inf_dist == 'norm':
        return torch.dist(inf_scores, uniform_scores, norm)
    elif inf_dist == 'kl':
        return torch.sum(uniform_scores * torch.log(torch.div(uniform_scores, inf_scores)))

    raise NotImplementedError(f'{inf_dist} is not implemented yet, please use "norm" or "kl"')


def _get_max_ent(inf_scores: torch.Tensor, norm: int) -> torch.Tensor:
    """
    Calculate the norm of 1 divided by the information
    :param inf_scores: tensor holding batch information scores
    :param norm: which norm to use
    :return:
    """
    return torch.norm(torch.div(1, inf_scores), p=norm)


def _get_max_ent_minus(inf_scores: torch.Tensor, norm: int) -> torch.Tensor:
    """
    Calculate -1 * the norm of the information
    :param inf_scores: tensor holding batch information scores
    :param norm: which norm to use
    :return:
    """
    return -1 * torch.norm(inf_scores, p=norm) + 0.1


def get_regularization_term(inf_scores: torch.Tensor, inf_dist: str, norm: int, optim_method: str) -> torch.Tensor:
    """
    Compute the regularization term given a batch of information scores
    :param inf_scores: tensor holding a batch of information scores
    :param inf_dist: defines which distance measure to use ("kl" or "norm")
    :param norm: defines which norm to use (1 or 2)
    :param optim_method: Define optimization method (possible methods: "min_ent", "max_ent", "max_ent_minus",
     "normalized")
    :return:
    """

    if optim_method == 'min_ent':
        return torch.norm(inf_scores, p=norm)
    elif optim_method == 'max_ent':
        return _get_max_ent(inf_scores, norm)
    elif optim_method == 'max_ent_minus':
        return _get_max_ent_minus(inf_scores, norm)
    elif optim_method == 'normalized':
        return _get_normalized_score(inf_scores, norm, inf_dist)

    raise NotImplementedError(f'"{optim_method}" is unknown')


def get_expanded_logits(logits: torch.Tensor, num_eval_samples: int, logits_flg: bool = True) -> torch.Tensor:
    """
    Perform Softmax and then expand the logits depends on the num_eval_samples
    :param logits_flg: whether the input is logits or softmax
    :param logits: tensor holds logits outputs from the model
    :param num_eval_samples: times to duplicate
    :return:
    """
    if logits_flg:
        logits = torch.nn.functional.softmax(logits, dim=1)
    expanded_logits = logits.repeat(1, num_eval_samples)

    return expanded_logits.view(expanded_logits.shape[0] * num_eval_samples, -1)
