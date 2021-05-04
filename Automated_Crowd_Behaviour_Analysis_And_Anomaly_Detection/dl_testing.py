from tensorflow.keras.preprocessing import image
from vggModel import vggModel
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import math
import os
from glob import glob
from scipy import stats as s
import matplotlib.pyplot as plt


def dl_testing():
    WEIGHTS_PATH = 'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/weights_test42.hdf5'
    PREDICT_BATCH_SIZE = 64

    RESULT_PATH = 'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/test.txt'

    # creating the model
    base_model, model = vggModel()

    # loading the trained weights
    model.load_weights(WEIGHTS_PATH)

    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # getting the test list
    f = open("Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Anomaly_Detection_splits/Test1.txt", "r")
    temp = f.read()
    videos = temp.split('\n')
    del temp

    # creating the dataframe
    test = pd.DataFrame()
    test['video_name'] = videos
    test = test[:-1]
    test_videos = test['video_name']
    test.head()

    # creating the tags
    classes = ['Anomaly', 'NoAnomaly']

    # creating evaluation lists
    predict = []
    actual = []
    truePositive = []
    falsePositive = []
    trueNegative = []
    falseNegative = []

    # for loop to extract frames from each test video
    for i in tqdm(range(test_videos.shape[0]), desc="Testing the Network"):
        count = 0
        videoFile = test_videos[i]
        cap = cv2.VideoCapture(
            'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Videos/' + videoFile)  # capturing the video from the given path
        frameRate = cap.get(5)  # frame rate

        # removing all other files from the temp folder
        files = glob('Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/temp/*')
        for f in files:
            os.remove(f)
        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frameId % math.floor(frameRate) == 0:
                # storing the frames of this particular video in temp folder
                filename = 'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/temp/' + "_frame%d.jpg" % count
                count += 1
                cv2.imwrite(filename, frame)
        cap.release()

        # reading all the frames from temp folder
        images = glob("Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/temp/*.jpg")

        # saving the frames to an array
        prediction_images = []
        for j in range(len(images)):
            img = image.load_img(images[j], target_size=None)
            img = image.img_to_array(img)
            img /= 255
            prediction_images.append(img)

        # converting all the frames for a test video into numpy array
        prediction_images = np.array(prediction_images)

        # extracting features using pre-trained model
        prediction_images = base_model.predict(prediction_images, batch_size=PREDICT_BATCH_SIZE)

        # converting features in one dimensional array
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 10 * 7 * 512)

        # predicting tags for each array
        prediction = model.predict_classes(prediction_images, batch_size=PREDICT_BATCH_SIZE)

        # appending the mode of predictions in predict list to assign the tag to the video
        predictionResult = classes[s.mode(prediction)[0][0]]
        predict.append(predictionResult)

        # creating a separate array of anomaly probabilities
        arrSlice = 0
        probability = []
        for j in range(len(prediction)):
            if prediction[j] == 0:
                probability.append(1.0)
                prediction[j] = 1
            elif prediction[j] == 1:
                probability.append(0.0)
                prediction[j] = 0
            if j != 0:
                probability[j] = sum(prediction[j - arrSlice:j + 1]) / len(prediction[j - arrSlice:j + 1])
            if arrSlice != 4:
                arrSlice += 1
        probability = np.array(probability)

        # creating a pyplot graph
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(640*px, 100*px))  # graph size in pixels
        plt.grid(True)
        # axes range
        plt.xlim(0, count * frameRate)
        plt.ylim(0, 1)
        plt.yticks([0, 0.5, 1])
        plt.ylabel('Anomaly\nProbability')
        count = 0

        # capturing the video
        cap = cv2.VideoCapture(
            'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Videos/' + videoFile)  # capturing the video from the given path
        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            # 1 every 30 frames is displayed
            if frameId % math.floor(frameRate) == 0:  # execute every frameRate frames
                plt.scatter(count * frameRate, probability[count], c='r')  # draw a dot on the graph
                fig.canvas.draw()
                count += 1
                plotImg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)  # convert the graph to numpy array
                plotImg = plotImg.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # reshape the graph
                cv2.putText(frame, "Frame " + str(int(frameId)), (10, 50), cv2.FONT_ITALIC, 0.65, (0, 0, 255), 2)
                frame = cv2.resize(frame, (640, 320))  # increase the size of the frame
                frame = np.vstack((frame, plotImg))  # combine the frame and the graph
                cv2.imshow('frame', frame)  # display the frame with the graph
                if cv2.waitKey(60) & 0xFF == ord('q'):  # USER CAN SKIP THE VIDEO BY CLICKING 'q'
                    break
        cap.release()
        cv2.destroyAllWindows()  # close all unnecessary windows

        # appending the actual tag of the video
        if videoFile.split('\\')[0] == "NoAnomaly":  # Negative
            actual.append("NoAnomaly")
            if predictionResult == actual[-1]:  # True Negative
                trueNegative.append(1)
                falseNegative.append(0)
            else:  # False Negative
                trueNegative.append(0)
                falseNegative.append(1)

        else:  # Positive
            actual.append("Anomaly")
            if predictionResult == actual[-1]:  # True Positive
                truePositive.append(1)
                falsePositive.append(0)
            else:  # False Positive
                truePositive.append(0)
                falsePositive.append(1)

    # checking the accuracy of the predicted tags
    with open(RESULT_PATH, 'w') as f:
        print("True Positive rate: %f" % (sum(truePositive) / len(truePositive) * 100), file=f)
        print("False Positive rate: %f" % (sum(falsePositive) / len(falsePositive) * 100), file=f)
        print("True Negative rate: %f" % (sum(trueNegative) / len(trueNegative) * 100), file=f)
        print("False Negative rate: %f" % (sum(falseNegative) / len(falseNegative) * 100), file=f)
        accuracy = 100 * ((sum(truePositive) + sum(trueNegative)) / (sum(truePositive) + sum(trueNegative) +
                                                                     sum(falsePositive) + sum(falseNegative)))
        print("Accuracy: %f" % accuracy, file=f)
