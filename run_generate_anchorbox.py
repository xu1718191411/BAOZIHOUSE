import numpy as np
import matplotlib.pyplot as plt


def generate_anchorbox(boxSize, scale, ratios):
    print(boxSize)
    print(scale)
    print(ratios)

    prevBoxsScales = np.tile(scale, (2, len(scale))).T
    prevBoxsScales = prevBoxsScales * boxSize
    print(prevBoxsScales)

    preBoxAreas = prevBoxsScales[:, 0] * prevBoxsScales[:, 1]
    print(preBoxAreas)

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


PYRAMID_LEVEL = [3, 4, 5, 6, 7]

TEST_SCALE = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
TEST_RATIO = np.array([0.5, 1, 2])

strides = [2 ** level for level in PYRAMID_LEVEL]
boxSizeBaseSizes = [2 ** (level + 2) for level in PYRAMID_LEVEL]
imageShape = (640, 586)


def shift_boxes(positionFixedAnchorBoxes, imageShape, stride, boxSizeBaseSize):
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


idx = 0
position_fixed_anchor_boxes = generate_anchorbox(boxSizeBaseSizes[0], TEST_SCALE, TEST_RATIO)
centerPositions, res = shift_boxes(position_fixed_anchor_boxes, imageShape, strides[idx], boxSizeBaseSizes[idx])

x = centerPositions[:, :, 0].ravel()
y = centerPositions[:, :, 1].ravel()
plt.plot(x, y, "o",markersize=0.3)
plt.show()

a = 1
