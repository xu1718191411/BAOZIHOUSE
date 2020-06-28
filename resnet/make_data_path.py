import os
def make_data_path(train_path, val_path, categories):
    types = os.listdir(train_path)
    trainData = []
    valData = []

    for type in types:
        typePath = os.path.join(train_path, type)
        filenames = os.listdir(typePath)

        for filename in filenames:
            filepath = os.path.join(typePath, filename)
            trainData.append({'filepath': filepath, 'category': categories.index(type)})

    for type in types:
        typePath = os.path.join(val_path, type)
        filenames = os.listdir(typePath)

        for filename in filenames:
            filepath = os.path.join(typePath, filename)
            valData.append({'filepath': filepath, 'category': categories.index(type)})


    return trainData,valData