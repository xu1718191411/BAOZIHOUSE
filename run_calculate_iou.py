import numpy as np
import torch
import matplotlib.pyplot as plt


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

    plt.show()

    calculateIOU(anchorBox, targets)


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

    anchorBoxesArea = torch.unsqueeze(anchorBoxesArea,dim=1)

    unionArea = anchorBoxesArea + targetArea - innerArea

    unionArea = torch.clamp(unionArea,min=1e-8)


    iou = innerArea / unionArea

    print(iou)


# cx1,cy1,cx2,cy2
anchorBoxes = torch.Tensor([
    [25, 18, 43, 27],
    [16, 2, 31, 17],
    [31, 16, 3, 35],
    [10, 25, 15, 21],
])

# cx1,cy1,cx2,cy2
targetBoxes = torch.Tensor([
    [20, 25, 35, 10],
    [8, 5, 12, 26],
])

run_calculate_iou(zhengze(anchorBoxes), zhengze(targetBoxes))
