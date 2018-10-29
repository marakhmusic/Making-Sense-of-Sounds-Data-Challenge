
import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset = pd.read_csv('audio_dataset.csv')
print(np.shape(dataset))
data = dataset.iloc[:, 2:42]


print(data.head())
print(data.shape)


# First center and scale the data
scaled_data = preprocessing.scale(data)
#scaled_data = StandardScaler().fit_transform(data)
print (scaled_data)
pca = PCA(n_components=10)  # create a PCA object
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


effects = ['effects' + str(i) for i in range(0, 300)]
human = ['human' + str(i) for i in range(0, 300)]
nature = ['nature' + str(i) for i in range(0, 300)]
music = ['music' + str(i) for i in range(0, 300)]
urban = ['urban' + str(i) for i in range(0, 300)]
pca_df = pd.DataFrame(pca_data, index=[*effects, *human, *nature, *music, *urban], columns=labels)
classes = dataset.iloc[:,-1].values
#print (z)
colors = np.array(["black", "green", "yellow", "red", "blue"])
clusters = np.array(["effects", "human", "nature","music", "urban"])
plt.scatter(pca_df.PC1, pca_df.PC2, c=colors[classes], label = True)
recs = []
for i in range(0,len(colors[classes]),300):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[classes][i]))
plt.legend(recs, clusters, loc = 'lower right')
plt.title('Sound Challenge')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))


#for sample in pca_df.index:
 #   plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

#plt.scatter(x,y, c=colors[z])
"""
xlabel = ('PC1 - {0}%'.format(per_var[0]))
ylabel = ('PC2 - {0}%'.format(per_var[1]))
zlabel = ('PC2 - {0}%'.format(per_var[2]))

ax.scatter(xlabel, ylabel, zlabel, c=c, marker=m)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.show()

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))



#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################

## get the name of the top 10 measurements (genes) that contribute
## most to pc1.
## first, get the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes)
## now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# get the names of the top 10 genes
top_10_genes = sorted_loading_scores[0:10].index.values

## print the gene names and their scores (and +/- sign)
print(loading_scores[top_10_genes]) < span
id = "mce_SELREST_start"
style = "overflow:hidden;line-height:0;" > < / span >
"""