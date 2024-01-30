#!/usr/bin/env python
# coding: utf-8

# ## MMAI 5500 Assignment 3

# #### Jiawen Li

# In[18]:


import os
import cv2


# #### Extract images from video and stored as image.JPEG

# In[2]:


def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
    """
    Converts the video file (assignment3_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
   # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')
    
    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # Instantiate the video object.
    video = cv2.VideoCapture(filename)
    
    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")
    
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()
    
    if i:
        print(f'Video converted\n{i} images written to {img_folder}')


# In[12]:


convert_video_to_images(img_folder='/Users/jiawenli/Desktop/MMAI5500_assignment3/img_folder', filename="assignment3_video.avi")


# ### Load the extracted images
# - Return the images as a Numpy array of flattened images
# - And also return a list of the resized images

# In[19]:


from PIL import Image
from glob import glob
import numpy as np


# In[20]:


def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.
    
    The images are returned both as a Numpy array of flattened images
    (i.e. the images with the 3-d shape (im_width, im_height, num_channels)
    are reshaped into the 1-d shape (im_width x im_height x num_channels))
    and a list with the images with their original number of dimensions
    suitable for display.
    
    Arguments
    ---------
    img_dir : (string) the directory where the images are stored.
    im_width : (int) The desired width of the image.
                     The default value works well.
    im_height : (int) The desired height of the image.
                      The default value works well.
    
    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized images.
    """
    
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()

    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()

    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    
    return X, images


# In[21]:


X, image = load_images(img_dir='/Users/jiawenli/Desktop/MMAI5500_assignment3/img_folder', im_width=60, im_height=44)


# In[11]:


X


# ### Build and Train an autoencoder for anomaly detection

# Build a Convolutional autoencoder for video frame data.

# In[8]:


pip install tensorflow


# #### Build the Convolutional Autoencoder architecture

# In[134]:


from tensorflow import keras
from keras import layers
from keras import backend as K

K.clear_session()

input_img = keras.Input(shape=(44, 60, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='valid')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# #### Reshape our input image X and train our autoencoder

# In[135]:


import numpy as np
import matplotlib.pyplot as plt

X = X.reshape(-1, 44, 60, 3)

autoencoder.fit(X, X,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X, X))


# #### Save the trained model

# In[172]:


autoencoder.save('trained_model.keras')


# #### Load the trained model

# In[173]:


from keras.models import load_model

autoencoder = load_model('trained_model.keras')


# Specify an anomalous image as an example from the imput data X and assign it to frame, and calculate the loss for that frame.

# In[159]:


frame_1 = X[704]
frame_2 = X[946]

frame_1 = frame_1.reshape((1, 44, 60, 3))
frame_2 = frame_2.reshape((1, 44, 60, 3))

loss_1 = autoencoder.evaluate(frame_1, frame_1, verbose=0)
loss_2 = autoencoder.evaluate(frame_2, frame_2, verbose=0)


# In[160]:


loss_1


# In[161]:


loss_2


# In[162]:


reconstructed = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructed, 2), axis=(1, 2, 3))


# - The predict() function generates the output that represents the model's best attempt at reconstructing the original input data X after it has been encoded and decoded within the autoencoder architecture.
# 
# - Calculate the Mean Squared Error (MSE) between the original data X and the reconstructed data reconstructed. The MSE is a measure of the quality of the reconstruction; it quantifies the difference between the original input X and the output produced by the autoencoder.

# In[163]:


plt.hist(mse, bins=50)
plt.xlabel("Test MSE loss")
plt.ylabel("Number of samples")
plt.show()


# Most of the images were reconstructed with a low error (0.001 to 0.002) and are likely "normal" images.
# 
# There is a clear gap between the "normal" MSE values and the anomalous ones, which are represented by a long tail to the right of the distribution. It appears that most of our data falls under an MSE loss of about 0.002, and then there's a long tail extending to higher MSE values. 
# 
# We want to be more aggresive, we want. to catch most of the anomalies, so we set the threshold to 0.002.

# #### Set the threshold as 0.002 based on the plot 

# ### Define the predict function

# In[23]:


def predict(frame):
    """
    Argument
    --------
    frame : Video frame with shape == (44, 60, 3) and dtype == float32.
    
    Return
    ------
    anomaly : A boolean indicating whether the frame is an anomaly or not.
    """
    threshold = 0.002
    
    frame = frame.reshape((1, 44, 60, 3))

    loss = autoencoder.evaluate(frame, frame, verbose=0)
    
    # If the loss exceeds the threshold, it's an anomaly
    anomaly = loss > threshold
    return anomaly


# ### Specify a frame from the data, use the predict function to detect whether the frame is anomalous or not

# In[25]:


frame = X[900]
predict(frame)


# #### Save the trained model

# In[12]:


autoencoder.save('trained_model.keras')


# #### Load the model

# In[22]:


from keras.models import load_model

autoencoder = load_model('trained_model.keras')


# In[ ]:





# In[ ]:





# In[ ]:




