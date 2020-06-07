import torch


def fun_focal_loss():
    alpha = 0.25
    gama = 2.00

    classificition = torch.Tensor([
        [0.001, 0.003, 0.92, 0.07, 0.006],
        [0.09, 0.15, 0.12, 0.09, 0.55],
        [0.398, 0.012, 0.239, 0.113, 0.238],
        [0.796, 0.008, 0.181, 0.003, 0.012],
        [0.116, 0.198, 0.234, 0.341, 0.111],
    ])

    teacher = torch.Tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    alphaValue = torch.ones(classificition.shape) * alpha

    alphaWeight = torch.where(torch.eq(teacher, 1), alphaValue, 1 - alphaValue)

    gamaWeight = torch.where(torch.eq(teacher, 1), 1 - classificition, classificition)
    gamaWeight = torch.pow(gamaWeight,gama)

    loss = -1 * teacher * torch.log(classificition) + (-1 * (1 - teacher) * (1 - classificition))

    loss = loss * alphaWeight * gamaWeight

    loss = loss.sum() / classificition.shape[0]
    print(loss)


fun_focal_loss()
