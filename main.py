import matplotlib.pylab as plt
from internal.dataHandler import handleTrainingData as trainData
from constants.credentials import credentials


# Setup of image directory and categories
train = trainData(credentials.CATEGORIES, credentials.DATADIR, credentials.IMGSIZE)
# Get training data
train.init_data()
# Shuffles dataset
train.shuffle(train.Data)
# Appends training data and reshapes
train.append()

# train.exportData(train.X,train.y)

plt.imshow(train.X[0],cmap="gray")
print(train.y[0])
plt.show()