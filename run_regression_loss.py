import torch

anchorBoxes = torch.Tensor([
    [1, 5, 17, 25],
    [22, 3, 38, 15],
    [4, 16, 22, 37],
    [23, 23, 38, 35],
])

# cx1,cy1,cx2,cy2
targetBoxes = torch.Tensor([
    [20, 15, 35, 3, 5],
    [5, 3, 16, 30, 3],
])

anchorBoxTargets = torch.Tensor(
    [
        [5., 3., 16., 30., 3.],
        [20., 3., 35., 15., 5.],
        [5., 3., 16., 30., 3.],
        [5., 3., 16., 30., 3.]
    ]
)

positiveIndexes = torch.Tensor([
    True, True, False, False
]).bool()

regression = torch.Tensor([
    [-0.0064, -0.0055, 0.095, 0.095],
    [0.0030, -0.0043, -0.0095, 0.0018],
    [0.0009, -0.0083, -0.0095, -0.0095],
    [-0.0036, 0.0023, -0.0056, -0.0068],
])


def calculateRegressionLoss(anchorBoxes, positiveIndexes, targetBoxes, regression):
    print(anchorBoxes)

    anchorBoxes = anchorBoxes[positiveIndexes, :]
    targetBoxes = targetBoxes[positiveIndexes, :]
    regression = regression[positiveIndexes, :]

    anchorBoxWidth = anchorBoxes[:, 2] - anchorBoxes[:, 0]
    anchorBoxHeight = anchorBoxes[:, 3] - anchorBoxes[:, 1]

    anchorBoxCenterX = anchorBoxes[:, 0] + 0.5 * anchorBoxWidth
    anchorBoxCenterY = anchorBoxes[:, 1] + 0.5 * anchorBoxHeight

    gtWidth = targetBoxes[:, 2] - targetBoxes[:, 0]
    gtHeight = targetBoxes[:, 3] - targetBoxes[:, 1]

    gtCenterX = targetBoxes[:, 0] + 0.5 * gtWidth
    gtCenterY = targetBoxes[:, 1] + 0.5 * gtHeight

    detaX = (gtCenterX - anchorBoxCenterX) / anchorBoxWidth
    detaY = (gtCenterY - anchorBoxCenterY) / anchorBoxHeight

    detaWidth = torch.log(gtWidth / anchorBoxWidth)
    detaHeight = torch.log(gtHeight / anchorBoxHeight)

    stacks = torch.stack((detaX, detaY, detaWidth, detaHeight))
    stacks = stacks.t()

    stacks = stacks / torch.Tensor([0.1, 0.1, 0.2, 0.2])
    print(stacks)

    diff = abs(regression - stacks)

    loss = torch.where(torch.le(diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(diff, 2), diff - 0.5 / 9.0)
    loss = loss.mean()
    return loss


# calculateDeta(anchorBoxes, positiveIndexes, anchorBoxTargets, regression)
