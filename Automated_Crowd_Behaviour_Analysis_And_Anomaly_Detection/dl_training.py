import numpy as np
import pandas as pd
from vggModel import vggModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_and_train():

    MODEL_CHECKPOINT_PATH = 'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/weights_test42.hdf5'
    PREDICT_BATCH_SIZE = 32

    # opening the file containing the list of training samples
    train = pd.read_csv('Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/train31.csv')
    train.head()

    # creating an empty list
    train_image = []

    # for loop to read and store frames
    for i in tqdm(range(train.shape[0]), desc="Loading training samples"):
        # loading the image and keeping the original dataset resolution as 240x320
        img = image.load_img('Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/test31/' + train['class'][i] + '-' +
                             train['image'][i], target_size=None)
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        img /= 255
        # appending the image to the train_image list
        train_image.append(img)

    # converting the list to numpy array
    train_image = np.array(train_image, dtype=np.float16)

    # checkpoint - displaying shape of the array
    print("Created numpy array of shape: ", train_image.shape)

    # separating the target
    train_class = train['class']

    # creating the training and validation set
    train_image, test_image, train_class, test_class = \
        train_test_split(train_image, train_class, random_state=42, test_size=0.20, stratify=train_class)

    # creating dummies of target variable for train and validation set
    train_class = pd.get_dummies(train_class)
    test_class = pd.get_dummies(test_class)

    # creating the model
    base_model, model = vggModel()

    # checkpoint - displaying shape of the training sample
    print("Split training sample of shape: ", train_image.shape)

    # extracting features for training frames
    train_image = base_model.predict(train_image, batch_size=PREDICT_BATCH_SIZE, verbose=1)

    # checkpoint - displaying shape of the predicted training sample
    print("Predicted training sample of shape: ", train_image.shape)

    # extracting features for validation frames
    test_image = base_model.predict(test_image, batch_size=PREDICT_BATCH_SIZE, verbose=1)

    # checkpoint - displaying shape of the predicted test sample
    print("Predicted test sample of shape: ", test_image.shape)

    # reshaping the training as well as validation frames in single dimension
    train_image = train_image.reshape((train_image.shape[0], 10 * 7 * 512))
    test_image = test_image.reshape((test_image.shape[0], 10 * 7 * 512))

    # normalizing the pixel values
    train_image /= np.amax(train_image)
    test_image /= np.amax(test_image)

    # checkpoint - displaying shape of the prepared sample
    print("Sample of shape ", train_image.shape, " is ready for training")
    base_model.summary()
    model.summary()

    # defining a function to save the weights of the best model
    mcp_save = ModelCheckpoint(MODEL_CHECKPOINT_PATH, save_best_only=True, monitor='loss', mode='min', verbose=1)

    # defining a function to avoid overfitting
    early_stop = EarlyStopping(monitor='loss', mode='min', patience=50, verbose=1)

    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training the model
    model.fit(train_image, train_class, epochs=300, validation_data=(test_image, test_class),
              callbacks=[mcp_save, early_stop], batch_size=1024, verbose=2)
