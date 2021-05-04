import math
from glob import glob
import cv2
import pandas as pd
from tqdm import tqdm


def videoSegmentation():

    # open the .txt file which have names of training videos
    f = open("Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Anomaly_Detection_splits/Train.txt", "r")
    temp = f.read()
    videos = temp.split('\n')

    # creating a dataframe having video names
    train = pd.DataFrame()
    train['video_name'] = videos
    train = train[:-1]
    train.head()

    # creating tags for training videos
    train_video_tag = []
    for i in tqdm(range(train.shape[0])):
        train_video_tag.append(train['video_name'][i].split('\\')[0])

    train['tag'] = train_video_tag

    # storing the frames from training videos
    for i in tqdm(range(train.shape[0])):
        count = 0
        videoFile = train['video_name'][i]
        cap = cv2.VideoCapture(
            'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Videos/' + train_video_tag[i] + '/'
            + videoFile.split(' ')[0].split('\\')[1])  # capturing the video from the given path
        frameRate = cap.get(5)  # frame rate
        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frameId % math.floor(frameRate) == 0:
                # storing the frames in a new folder named train_1
                filename = 'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/test31/' + \
                           videoFile.split('\\')[0] + "-" + \
                           videoFile.split('\\')[1].split(' ')[0] + "_frame%d.jpg" % count
                count += 1
                cv2.imwrite(filename, frame)
        cap.release()


def taggingFrames():
    # getting the names of all the images
    images = glob("Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/test31/*.jpg")
    train_image = []
    train_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        train_image.append((images[i].split('\\')[1]).split('-')[1])
        # print(images[i].split('\\')[1])
        train_class.append((images[i].split('\\')[1]).split('-')[0])
        # print(images[i].split('\\')[0])

    # storing the images and their class in a dataframe
    train_data = pd.DataFrame()
    train_data['image'] = train_image
    train_data['class'] = train_class

    # converting the dataframe into csv file
    train_data.to_csv('Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/train31.csv', header=True, index=False)
