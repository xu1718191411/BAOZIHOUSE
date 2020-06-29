import torch


def collater(data):

    widths = []
    heights = []
    for d in data:
        widths.append(d[0].shape[2])
        heights.append(d[0].shape[1])
    batchSize = len(data)
    maxWidth = max(widths)
    maxHeight = max(heights)
    finalImages = torch.zeros((batchSize, 3, maxHeight, maxWidth))
    finalCategories = torch.zeros(batchSize).long()
    for batch_index, d in enumerate(data):
        img = d[0]
        category = d[1]
        finalImages[batch_index, :img.shape[0], :img.shape[1], :img.shape[2]] = img
        finalCategories[batch_index] = category

    return {'images': finalImages, 'categories': finalCategories}
