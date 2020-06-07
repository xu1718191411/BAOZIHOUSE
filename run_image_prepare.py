import skimage
import numpy as np
import skimage.io
import skimage.transform


def test_image_flip(path):
    image = skimage.io.imread(path)  # (H,W,C)

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


def test_normalization(path):
    image = skimage.io.imread(path)  # (H,W,C)

    image = image.astype(np.float32) / 255.0
    skimage.io.imshow(image)
    skimage.io.show()

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    transformedImage = (image.astype(np.float32) - mean) / std

    skimage.io.imshow(transformedImage)
    skimage.io.show()


def test_resize(path):
    smallerLineMaxLength = 608
    largerLineMaxLength = 1024

    image = skimage.io.imread(path)

    skimage.io.imshow(image)
    skimage.io.show()

    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    matchMin = min(imageWidth, imageHeight)
    matchMax = max(imageWidth, imageHeight)

    scale = smallerLineMaxLength / matchMin

    resizedMax = int(matchMax * scale)

    if resizedMax > largerLineMaxLength:
        scale = largerLineMaxLength / matchMax

    resizedWidth = int(imageWidth * scale)
    resizedHeight = int(imageHeight * scale)
    trasformedImage = skimage.transform.resize(image, (resizedHeight, resizedWidth))

    skimage.io.imshow(trasformedImage)
    skimage.io.show()

    paddingHeight = 32 - int(resizedHeight) % 32
    paddingWidth = 32 - int(resizedWidth) % 32

    newImageStruc = np.zeros((resizedHeight + paddingHeight, resizedWidth + paddingWidth, 3))
    print(newImageStruc.shape)
    newImageStruc[:resizedHeight, :resizedWidth, :] = trasformedImage

    skimage.io.imshow(newImageStruc)
    skimage.io.show()


test_resize("./test.jpg")

# test_image_flip("./test.jpg")
