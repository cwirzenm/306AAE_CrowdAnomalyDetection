import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # turning off all INFO and WARNING messages
from gpuTest import usingGPU
from videoSegmentation import videoSegmentation, taggingFrames
from dl_training import read_and_train
from dl_testing import dl_testing

""" 

This code is an improved, extended and adapted version of the code that can be found on this tutorial.
https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/

Adaptations include:
 - GPU processing
 - video segmenting and frame tagging
 - higher sample resolution
 - different model architecture
 - model evaluation

Improvements include:
 - processing speed and memory management
 - model fitting
 - accuracy metrics
 - anomaly probability graph
 - overall readability of the code
 
For the specific list and in depth explanation of adaptations and improvements, 
                                                        please refer to the appropriate section of the final report.

"""

if __name__ == '__main__':

    usingGPU()
    # videoSegmentation()
    # taggingFrames()
    # read_and_train()
    dl_testing()
