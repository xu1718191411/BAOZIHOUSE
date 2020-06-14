import torch


def fun_focal_loss(classificition, teacher):
    alpha = 0.25
    gama = 2.00

    alphaValue = torch.ones(classificition.shape) * alpha

    alphaWeight = torch.where(torch.eq(teacher, 1), alphaValue, 1 - alphaValue)

    gamaWeight = torch.where(torch.eq(teacher, 1), 1 - classificition, classificition)
    gamaWeight = torch.pow(gamaWeight, gama)

    loss = -1 * teacher * torch.log(classificition) + (-1 * (1 - teacher) * torch.log(1 - classificition))

    loss = loss * alphaWeight * gamaWeight

    loss = loss.sum() / classificition.shape[0]
    return loss


# classificition = torch.Tensor([
#     [0.001, 0.003, 0.92, 0.07, 0.006],
#     [0.09, 0.15, 0.12, 0.09, 0.55],
#     [0.398, 0.012, 0.239, 0.113, 0.238],
#     [0.796, 0.008, 0.181, 0.003, 0.012],
#     [0.116, 0.198, 0.234, 0.341, 0.111],
# ])
#
# teacher = torch.Tensor([
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

# classification = torch.Tensor([
#     [0.053, 0.2463, 0.753, 0.2352, 0.3413, 0.1234, 0.3253, 0.235, 0.235, 0.0023],
#     [0.235, 0.3435, 0.023, 0.0023, 0.7897, 0.23453, 0.5235, 0.5623, 0.3423, 0.3462],
#     [0.0023, 0.3452, 0.3423, 0.0023, 0.9235, 0.234, 0.0023, 0.1242, 0.0023, 0.1235],
#     [0.0023, 0.124, 0.0235, 0.1252, 0.03252, 0.0023, 0.2353, 0.2352, 0.135, 0.2352]
# ])
#
# teacher = torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#                         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
#
# loss = fun_focal_loss(classification, teacher)

# print(loss)
