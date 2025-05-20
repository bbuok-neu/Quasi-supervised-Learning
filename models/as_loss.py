import torch


def approximately_supervised_loss_exp(none_reduction_loss, gamma=1.0):

    """
    :param none_reduction_loss: receive a loss tensor with reduction='none', Shape [B, C, H, W], where B is batch size.
    :param gamma: as loss gamma parameter.
    :return: scalar as loss tensor.
    """
    if none_reduction_loss.dim() == 0:
        raise ValueError(f"Input 'none_reduction_loss' must not be a scalar. Got shape: {none_reduction_loss.shape}")
    elif none_reduction_loss.shape[0] == 0:
        raise ValueError(f"Input 'none_reduction_loss' batch size must not be 0. Got shape: {none_reduction_loss.shape}")
    else:
        # loss = 0
        # for dimi in range(none_reduction_loss.shape[0]):
        #     loi = torch.mean(none_reduction_loss[dimi])
        #     loss += torch.exp(-loi / gamma)
        # loss /= none_reduction_loss.shape[0]
        # loss = -torch.log(loss)
        mean_loss_per_sample = none_reduction_loss.view(none_reduction_loss.shape[0], -1).mean(dim=1)  # shape: [batch_size]
        exp_term = torch.exp(-mean_loss_per_sample / gamma)  # shape: [batch_size]
        loss = -torch.log(exp_term.mean())

    return loss


def approximately_supervised_loss_power(none_reduction_loss, m=1.2):
    """
    :param none_reduction_loss: receive a loss tensor with reduction='none', Shape [B, C, H, W], where B is batch size.
    :param gamma: as loss gamma parameter.
    :return: scalar as loss tensor.
    """
    if none_reduction_loss.dim() == 0:
        raise ValueError(f"Input 'none_reduction_loss' must not be a scalar. Got shape: {none_reduction_loss.shape}")
    elif none_reduction_loss.shape[0] == 0:
        raise ValueError(f"Input 'none_reduction_loss' batch size must not be 0. Got shape: {none_reduction_loss.shape}")
    else:
        mean_loss_per_sample = none_reduction_loss.view(none_reduction_loss.shape[0], -1).mean(
            dim=1)  # shape: [batch_size]
        loss = (torch.sum(mean_loss_per_sample ** (1 / (1 - m))) / none_reduction_loss.shape[0]) ** (1 - m)
    return loss