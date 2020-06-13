import numpy as np
import torch
import matplotlib.pyplot as plt

from run_focal_loss import fun_focal_loss


def zhengze(annotations):
    minX = torch.min(annotations[:, 0], annotations[:, 2])
    maxX = torch.max(annotations[:, 0], annotations[:, 2])

    minY = torch.min(annotations[:, 1], annotations[:, 3])
    maxY = torch.max(annotations[:, 1], annotations[:, 3])

    result = torch.zeros(annotations.shape)

    result[:, 0] = minX
    result[:, 1] = minY
    result[:, 2] = maxX
    result[:, 3] = maxY

    size = annotations.shape
    if size[1] > 4:
        result[:, 4:] = annotations[:, 4:]

    return result


def run_calculate_iou(anchorBox, targets):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for box in anchorBox.numpy():
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        cx = 0
        cy = 0
        width = x2 - x1
        height = y2 - y1
        plt.plot(cx, cy, "o")
        rect = plt.Rectangle([x1, y1], width, height, fill=None)
        ax.add_patch(rect)

    for box in targets.numpy():
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        cx = 0
        cy = 0
        width = x2 - x1
        height = y2 - y1
        plt.plot(cx, cy, "o", "red")
        rect = plt.Rectangle([x1, y1], width, height, fill=True)
        ax.add_patch(rect)
    #
    plt.show()

    return calculateIOU(anchorBox, targets)


def calculateIOU(anchorBox, targets):
    cx1 = anchorBox[:, 0]
    cy1 = anchorBox[:, 1]
    cx2 = anchorBox[:, 2]
    cy2 = anchorBox[:, 3]

    cx3 = targets[:, 0]
    cy3 = targets[:, 1]
    cx4 = targets[:, 2]
    cy4 = targets[:, 3]

    cx1 = torch.unsqueeze(cx1, dim=1)
    cy1 = torch.unsqueeze(cy1, dim=1)
    cx2 = torch.unsqueeze(cx2, dim=1)
    cy2 = torch.unsqueeze(cy2, dim=1)

    iouW = torch.clamp(torch.min(cx2, cx4) - torch.max(cx1, cx3), min=0)
    iouH = torch.clamp(torch.min(cy2, cy4) - torch.max(cy1, cy3), min=0)

    innerArea = iouW * iouH

    anchorBoxesArea = (anchorBox[:, 0] - anchorBox[:, 2]) * (anchorBox[:, 1] - anchorBox[:, 3])

    targetArea = (cx3 - cx4) * (cy3 - cy4)

    anchorBoxesArea = torch.unsqueeze(anchorBoxesArea, dim=1)

    unionArea = anchorBoxesArea + targetArea - innerArea

    unionArea = torch.clamp(unionArea, min=1e-8)

    iou = innerArea / unionArea

    print(iou)

    iouMaxValue, iouMaxIndexes = torch.max(iou, dim=1)

    negativeIndexes = torch.le(iouMaxValue, 0.4)

    classificationNum = 10
    anchorBoxsNum = anchorBox.shape[0]

    result = torch.ones((anchorBoxsNum, classificationNum))

    result[negativeIndexes] = 0

    positiveIndexes = torch.gt(iouMaxValue, 0.5)

    result[positiveIndexes] = 0

    anchorBoxTargets = targets[:][iouMaxIndexes]
    # anchorBoxTargets = targets[iouMaxIndexes, :]
    anchorBoxTargetsClassIndexes = anchorBoxTargets[positiveIndexes, 4].long() - 1
    result[positiveIndexes, anchorBoxTargetsClassIndexes] = 1
    return result


# cx1,cy1,cx2,cy2
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

teacher = run_calculate_iou(zhengze(anchorBoxes), zhengze(targetBoxes))

classification = torch.Tensor([
    [0.053,	0.2463,	0.753,	0.2352,	0.3413,	0.1234,	0.3253,	0.235,	0.235,	0.0023],
    [0.235,	0.3435,	0.023,	0.0023,	0.7897,	0.23453,	0.5235,	0.5623,	0.3423,	0.3462],
    [0.0023,	0.3452,	0.3423,	0.0023,	0.9235,	0.234,	0.0023,	0.1242,	0.0023,	0.1235],
    [0.0023,	0.124,	0.0235,	0.1252,	0.03252,	0.0023,	0.2353,	0.2352,	0.135,	0.2352]
])

print(teacher)
loss = fun_focal_loss(classification, teacher)
print(loss)



