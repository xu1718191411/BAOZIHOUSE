import skimage
import numpy as np
import skimage.io
import skimage.transform


def test_image_flip():
    image = skimage.io.imread("./test.jpg")  # (H,W,C)

    b = image.astype(np.float32) / 255.0
    skimage.io.imshow(b)
    skimage.io.show()

    c = b[:, ::-1, :]
    skimage.io.imshow(c)
    skimage.io.show()


def test_annotation_flip():
    imageWidth = 768

    annotations = np.array([
        [23, 56, 180, 195, 19],
        [54, 23, 89, 273, 8]
    ])

    print(annotations)

    x1 = annotations[:, 0]
    x2 = annotations[:, 2]

    flip_x1 = imageWidth - x1
    flip_x2 = imageWidth - x2

    annotations[:, 0] = flip_x1
    annotations[:, 2] = flip_x2

    print(annotations)


def test_normalization():
    image = np.array([
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ]
    ])
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    transformedImage = image.astype(np.float32) - mean / std
    print(transformedImage)


def test_resize(path):
    minWidth = 608
    minHeight = 1024

    image = skimage.io.imread(path)

    skimage.io.imshow(image)
    skimage.io.show()

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    scale = imageWidth / minWidth
    resizedWidth = int(minWidth)
    resizedHeight = int(imageHeight * scale)

    if resizedHeight > minHeight:
        scale = imageHeight / minHeight
        resizedHeight = int(minHeight)
        resizedWidth = int(imageWidth * scale)

    trasformedImage = skimage.transform.resize(image, (resizedHeight, resizedWidth))

    skimage.io.imshow(trasformedImage)
    skimage.io.show()

    paddingHeight = 32 - int(resizedHeight) % 32
    paddingWidth = 32 - int(resizedWidth) % 32

    newImageStruc = np.zeros((resizedHeight + paddingHeight, resizedWidth + paddingWidth,3))
    print(newImageStruc.shape)
    newImageStruc[:resizedHeight, :resizedWidth, :] = trasformedImage

    skimage.io.imshow(newImageStruc)
    skimage.io.show()

test_resize("./test.jpg")
