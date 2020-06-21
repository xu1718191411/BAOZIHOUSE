import numpy as np
import torch
from torchvision.ops import nms, box_iou
import matplotlib.pyplot as plt
import random

PYRAMID_LEVEL = [3, 4, 5, 6, 7]

TEST_SCALE = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
TEST_RATIO = np.array([0.5, 1, 2])

classNum = 10


def clip_anchor_boxes(transformedAnchorBoxes, w, h):
    print(transformedAnchorBoxes)

    cx = transformedAnchorBoxes[:, :, 0]
    cy = transformedAnchorBoxes[:, :, 1]
    width = transformedAnchorBoxes[:, :, 2]
    height = transformedAnchorBoxes[:, :, 3]

    torch.clamp(cx, min=0)
    torch.clamp(cy, min=0)
    torch.clamp(width, max=w)
    torch.clamp(height, max=h)

    result = torch.stack((cx, cy, width, height), dim=2)
    return result


def anchor_box_transform_with_regression(anchorBoxes, regressions):
    anchorBoxesWidth = anchorBoxes[:, :, 2] - anchorBoxes[:, :, 0]
    anchorBoxesHeight = anchorBoxes[:, :, 3] - anchorBoxes[:, :, 1]

    anchorBoxesCenterX = anchorBoxes[:, :, 0] + anchorBoxesWidth * 0.5
    anchorBoxesCenterY = anchorBoxes[:, :, 1] + anchorBoxesHeight * 0.5

    transformedCenterX = anchorBoxesCenterX + regressions[:, :, 0] * anchorBoxesWidth
    transformedCenterY = anchorBoxesCenterY + regressions[:, :, 1] + anchorBoxesHeight
    transformedWidth = torch.exp(regressions[:, :, 2]) * anchorBoxesWidth
    transformedHeight = torch.exp(regressions[:, :, 3]) * anchorBoxesHeight

    transformedStartX = transformedCenterX - 0.5 * transformedWidth
    transformedStartY = transformedCenterY - 0.5 * transformedHeight
    result = torch.stack((transformedStartX, transformedStartY, transformedWidth, transformedHeight), dim=2)

    return result


def generate_anchorbox(boxSize, scale, ratios):
    prevBoxsScales = np.tile(scale, (2, len(scale))).T
    prevBoxsScales = prevBoxsScales * boxSize

    preBoxAreas = prevBoxsScales[:, 0] * prevBoxsScales[:, 1]

    # w * h = area
    # w * w*ratio = area
    preBoxRatios = np.repeat(ratios, len(scale))
    preBoxW = np.sqrt(preBoxAreas / preBoxRatios)
    preBoxH = preBoxW * preBoxRatios

    anchorBox = np.zeros((len(scale) * len(ratios), 4))

    anchorBox[:, 2] = preBoxW
    anchorBox[:, 3] = preBoxH

    #
    anchorBox[:, 0::2] -= np.tile(anchorBox[:, 2] * 0.5, (2, 1)).T
    anchorBox[:, 1::2] -= np.tile(anchorBox[:, 3] * 0.5, (2, 1)).T
    return anchorBox


def shift_boxes(positionFixedAnchorBoxes, imageShape, stride):
    imageWidth = imageShape[1]
    imageHeight = imageShape[0]

    featuresWidth = int((imageWidth + 0.5 * stride) / stride)
    featureHeight = int((imageHeight + 0.5 * stride) / stride)

    featureXCoordinates = np.arange(0, featuresWidth) + 0.5
    featureYCoordinates = np.arange(0, featureHeight) + 0.5

    featureXCoordinates = featureXCoordinates * stride
    featureYCoordinates = featureYCoordinates * stride

    a, b = np.meshgrid(featureXCoordinates, featureYCoordinates)
    m = np.vstack((a.ravel(), b.ravel(), a.ravel(), b.ravel()))
    m = m.transpose()

    positionFixedAnchorBoxes = np.expand_dims(positionFixedAnchorBoxes, 0)
    m = np.expand_dims(m, 1)

    res = m + positionFixedAnchorBoxes

    return m[:, :, :2], res


strides = [2 ** level for level in PYRAMID_LEVEL]
boxSizeBaseSizes = [2 ** (level + 2) for level in PYRAMID_LEVEL]
imageShape = (640, 832)
idx = 4
boxSizeBaseSizes[idx]

position_fixed_anchor_boxes = generate_anchorbox(boxSizeBaseSizes[idx], TEST_SCALE, TEST_RATIO)
centerPositions, anchorBoxes = shift_boxes(position_fixed_anchor_boxes, imageShape, strides[idx])

featuresWidth = int((imageShape[1] + 0.5 * strides[idx]) / strides[idx])
featureHeight = int((imageShape[0] + 0.5 * strides[idx]) / strides[idx])

anchorBoxes = torch.from_numpy(anchorBoxes)

classifications = torch.rand((featuresWidth * featureHeight, 9, classNum)) * 0.01

regressions = torch.rand((featuresWidth * featureHeight, 9, 4))

transformedAnchorBoxes = anchor_box_transform_with_regression(anchorBoxes, regressions)

transformedAnchorBoxes = clip_anchor_boxes(transformedAnchorBoxes,imageShape[1],imageShape[0])

anchorBoxes = transformedAnchorBoxes.view(-1, 4)
classifications = classifications.view(-1, classNum)

targetNum = 3
targetFrequent = (featuresWidth * featureHeight * 9) // 3
for i in range(featuresWidth * featureHeight * 9):
    if i % targetFrequent == 0:
        classifications[i] = 0.9423 + random.random()

regressions = regressions.view(-1, 4)

scores, scoreIndexes = torch.max(classifications, dim=1)

validIndex = scores > 0.02

anchorBoxes = anchorBoxes[validIndex]
scoreIndexes = scoreIndexes[validIndex]
scores = scores[validIndex].double()

leftIndexes = nms(anchorBoxes, scores, iou_threshold=0.5)

finalBoxes = anchorBoxes[leftIndexes]

fig = plt.figure()
ax = fig.add_subplot(111)

for box in finalBoxes:
    width = box[2]
    height = box[3]
    cx = box[0] - 0.5 * width
    cy = box[1] - 0.5 * height

    rect = plt.Rectangle([cx, cy], width, height, fill=None)
    ax.add_patch(rect)

plt.xlim(-1200, 1900)
plt.ylim(-1200, 1900)
plt.show()
