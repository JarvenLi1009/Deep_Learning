# Deep_Learning_in_Business
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Candy_counter.ipynb
This is an object detection assignment.
In this file, I started by manually annotating the candy images among the 11 images, for a total of eight different types of candy. 
I then loaded and processed the annotations in COCO format into a dictionary, then converted this data into a Huggingface-compatible format and saved it as a JSON file. 
Next, I organized the folder structure containing the images and metadata.jsonl and loaded the data into Huggingface's DatasetDict for manipulation. In the data split, I selected some of the images as training and test data. 
I then preprocessed this data and trained it using the DETR model, a Transformer-based object detection technique.
After training, I saved the model and evaluated the performance using mean average precision (mAP). Ultimately, I applied this trained model for object detection for counting the number of candies and discriminating the type of candies.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Anomaly_Detection.py
In this project, I developed an anomaly detection system based on a self-encoder model designed to address key challenges in the field such as cyber security and surveillance. 
I processed the video data using Python and OpenCV to extract individual frames from the video dataset and convert them to JPEG images to store in local folder to prepare the data for model training. 
Then, I trained an auto-encoder using Keras which is a deep learning method to encode video frames, then I used reconstruction loss as a metric for identifying abnormal frames, and set a loss threshold based on reconstructuion loss to effectively differentiate between normal and abnormal frames to capture suspicious activities. Since the loss threshold for catching the abnormal frames is based on a personal opinion, the aggressiveness of the model would depend on the user's preference.
Additionally, I created a Python function to integrate the model for predicting anomalies in new video frames, which analyzes the frames based on the trained model and returns a Boolean value to indicate whether or not an anomaly exists in the corresponding image.
