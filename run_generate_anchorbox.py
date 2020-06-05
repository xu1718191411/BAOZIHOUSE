import numpy as np
import matplotlib.pyplot as plt

PYRAMID_LEVEL = [3, 4, 5, 6, 7]

TEST_SCALE = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
TEST_RATIO = np.array([0.5, 1, 2])

strides = [2 ** level for level in PYRAMID_LEVEL]
boxSizeBaseSizes = [2 ** (level + 2) for level in PYRAMID_LEVEL]
imageShape = (640, 832)

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


def run_generate_anchorbox():
    idx = 0
    anchorBoxSizes = generate_anchorbox(boxSizeBaseSizes[idx], TEST_SCALE, TEST_RATIO)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    for box in anchorBoxSizes:
        print(box)
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
    plt.xlim(-imageShape[0]/(2**PYRAMID_LEVEL[idx]), imageShape[0]/(2**PYRAMID_LEVEL[idx]))
    plt.ylim(-imageShape[1]/(2**PYRAMID_LEVEL[idx]), imageShape[1]/(2**PYRAMID_LEVEL[idx]))
    plt.show()

# run_generate_anchorbox()


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


def run_shift_boxes():
    idx = 4
    position_fixed_anchor_boxes = generate_anchorbox(boxSizeBaseSizes[idx], TEST_SCALE, TEST_RATIO)
    centerPositions, transformed_anchor_boxes = shift_boxes(position_fixed_anchor_boxes, imageShape, strides[idx],
                                                            boxSizeBaseSizes[idx])

    x = centerPositions[:, :, 0].ravel()
    y = centerPositions[:, :, 1].ravel()
    plt.plot(x, y, "o", markersize=0.3, markerfacecolor='black')
    plt.show()

    fig = plt.figure()
    x = centerPositions[:, :, 0].ravel()
    y = centerPositions[:, :, 1].ravel()
    plt.plot(x, y, "o", markersize=0.3, markerfacecolor='black')

    ax = fig.add_subplot(111)

    totalFeatureBoxNum = centerPositions.shape[0]
    testPoint1 = int(totalFeatureBoxNum*(0.056))
    testPoint2 = int(totalFeatureBoxNum*(0.257))
    testPoint3 = int(totalFeatureBoxNum*(0.395))
    testPoint4 = int(totalFeatureBoxNum*(0.689))
    testPoint5 = int(totalFeatureBoxNum*(0.903))
    sample_anchor_points = [testPoint1,testPoint2,testPoint3,testPoint4,testPoint5]

    for sample_point_index in sample_anchor_points:
        for i in range(9):
            x1 = transformed_anchor_boxes[sample_point_index][i][0]
            y1 = transformed_anchor_boxes[sample_point_index][i][1]
            x2 = transformed_anchor_boxes[sample_point_index][i][2]
            y2 = transformed_anchor_boxes[sample_point_index][i][3]
            center = centerPositions[sample_point_index]
            cx = center.ravel()[0]
            cy = center.ravel()[1]
            width = x2 - x1
            height = y2 - y1

            plt.plot(cx, cy, "o")
            rect = plt.Rectangle([x1, y1], width, height, fill=None)
            ax.add_patch(rect)

    plt.xlim(-200, 900)
    plt.ylim(-200, 900)
    plt.show()


run_shift_boxes()