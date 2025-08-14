import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []
classes = 43
dataset_dir = "C:/Users/Priya/OneDrive/Priya_onedrive/OneDrive/Desktop/AI"
#Retrieving the images and their labels 
metaDf = pd.read_csv(r"C:\Users\Priya\OneDrive\Priya_onedrive\OneDrive\Desktop\AI\Meta.csv") 
trainDf = pd.read_csv(r"C:\Users\Priya\OneDrive\Priya_onedrive\OneDrive\Desktop\AI\Test.csv") 
testDf= pd.read_csv(r"C:\Users\Priya\OneDrive\Priya_onedrive\OneDrive\Desktop\AI\Train.csv") 
labels = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h', '120 km/h', 'No overtaking',
               'No overtaking for tracks', 'Crossroad with secondary way', 'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock',
               'Other dangerous', 'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road', 'Roadwork', 'Traffic light',
               'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits', 'Only right', 'Only left', 'Only straight', 'Only straight and right', 
               'Only straight and left', 'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']
print('SHAPE of training set:',trainDf.shape)
print('SHAPE of test set:',trainDf.shape)
print('SHAPE of MetaInfo:',trainDf.shape)
trainDf['Path'] = list(map(lambda x: os.path.join(dataset_dir,x.lower()), trainDf['Path']))
testDf['Path'] = list(map(lambda x: os.path.join(dataset_dir,x.lower()), testDf['Path']))
metaDf['Path'] = list(map(lambda x: os.path.join(dataset_dir,x.lower()), metaDf['Path']))
trainDf.sample(10)
metaDf.sample(10)
print(trainDf.sample(10))
print(metaDf.sample(10))


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(25, 6))

# Train distribution
sns.countplot(data=trainDf, x='ClassId', ax=axs[0], palette="Set1")
axs[0].set_title('Train classes distribution')
axs[0].set_xlabel('Class ID')
axs[0].set_ylabel('Count')

# Test distribution
sns.countplot(data=testDf, x='ClassId', ax=axs[1], palette="Set2")
axs[1].set_title('Test classes distribution')
axs[1].set_xlabel('Class ID')
axs[1].set_ylabel('Count')
#plt.show()



trainDfDpiSubset = trainDf[(trainDf.Width < 80) & (trainDf.Height < 80)]
testDfDpiSubset = testDf[(testDf.Width < 80) & (testDf.Height < 80)]

g = sns.JointGrid(x="Width", y="Height", data=trainDfDpiSubset)

sns.kdeplot(data=trainDfDpiSubset, x="Width", y="Height", cmap="Reds", fill=False, ax=g.ax_joint)
sns.kdeplot(data=testDfDpiSubset, x="Width", y="Height", cmap="Blues", fill=False, ax=g.ax_joint)

g.fig.set_figwidth(25)
g.fig.set_figheight(8)
#plt.show()

#this line is if code is stuck after set2
#plt.show(block=False)

# Make sure dataset_dir points to your folder
dataset_dir = r"C:/Users/Priya/OneDrive/Priya_onedrive/OneDrive/Desktop/AI"

# Ensure full paths
metaDf["Path"] = metaDf["Path"].apply(lambda x: os.path.join(dataset_dir, x))

# Sort by ClassId
metaDf = metaDf.sort_values(by=['ClassId'])

sns.set_style("white")
rows, cols = 6, 8
fig, axs = plt.subplots(rows, cols, figsize=(25, 12))
plt.subplots_adjust(top=0.9)

idx = 0
for i in range(rows):
    for j in range(cols):
        if idx >= len(metaDf) or idx > 42:  # 43 classes max
            axs[i, j].axis('off')
            continue
        
        img_path = metaDf["Path"].iloc[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            axs[i, j].axis('off')
            idx += 1
            continue
        
        # Handle transparency if present
        if img.shape[2] == 4:
            img[np.where(img[:, :, 3] == 0)] = [255, 255, 255, 255]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (60, 60))
        
        axs[i, j].imshow(img)
        axs[i, j].set_title(labels[int(metaDf["ClassId"].iloc[idx])], fontsize=8)
        axs[i, j].axis('off')
        
        idx += 1

#plt.show()


rows = 10
cols = 10
fig, axs = plt.subplots(rows, cols, figsize=(25, 12))
plt.subplots_adjust(top=0.9, hspace=0.2)

cur_path = r"C:/Users/Priya/OneDrive/Priya_onedrive/OneDrive/Desktop/AI"
idx = 0

for i in range(rows):
    for j in range(cols):
        path = os.path.join(cur_path, trainDf["Path"].iloc[idx])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"⚠ Could not read image: {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (60, 60))

        axs[i, j].imshow(img)
        axs[i, j].set_title(labels[int(trainDf["ClassId"].iloc[idx])], fontsize=8)
        axs[i, j].axis("off")
        idx += 1

plt.show()
