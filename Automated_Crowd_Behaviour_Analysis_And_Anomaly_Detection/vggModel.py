from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def vggModel():

    # creating the base model of pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(240, 320, 3))
    base_model._make_predict_function()

    # defining the top architecture
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(35840,)))
    model.add(Dropout(0.8))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return base_model, model
