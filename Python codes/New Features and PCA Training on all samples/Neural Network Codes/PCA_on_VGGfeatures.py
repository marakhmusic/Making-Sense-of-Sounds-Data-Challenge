import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt  # NOTE: This was tested with matplotlib v. 2.1.0
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

dataset = np.load(r'/Users/mansoor/Georgia Tech/Sem I/MUSI7100/MUSI 7100 Project/Python codes/New Features and PCA Training on all samples/VGG Features/vgg_features_postprocessed.npy')
print(np.shape(dataset))
data = pd.DataFrame(dataset)

l = np.shape(data)
number = int((l[0]/5))
y = np.repeat(np.arange(5), number)
print(np.shape(y))

print(data.head())
print(data.shape)


# First center and scale the data
scaled_data = preprocessing.scale(data)
#scaled_data = StandardScaler().fit_transform(data)
print (scaled_data)
pca = PCA()  # create a PCA object
pca.fit(scaled_data)  # do the math
pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data



# The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]


plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# the following code makes a fancy looking plot using PC1 and PC2
effects = ['effects' + str(i) for i in range(0, 1500)]
human = ['human' + str(i) for i in range(0, 1500)]
nature = ['nature' + str(i) for i in range(0, 1500)]
music = ['music' + str(i) for i in range(0, 1500)]
urban = ['urban' + str(i) for i in range(0, 1500)]
pca_df = pd.DataFrame(pca_data, index=[*effects, *human, *nature, *music, *urban], columns=labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

X = pca_df.PC1
Y = pca_df.PC2
Z = pca_df.PC3
classes = y
colors = np.array(["black", "green", "yellow" , "red", "blue"])
clusters = np.array(["effects", "human", "nature","music", "urban"])
ax.scatter(X,Y,Z, c = colors[classes])
recs = []
for i in range(0,len(colors[classes]),1500):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[classes][i]))
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
plt.legend(recs, clusters, loc = 'lower right')
plt.show()


