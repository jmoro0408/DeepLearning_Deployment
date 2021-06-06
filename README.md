# Deep Learning Deployment with Docker and AWS

Small project to increase my knowledge of Docker and AWS deployment. 
The project uses a simple convolutional neural network (CNN) to make predictions on some simple voice commands ("up", "down", "right", "left" and others). The code follows a "The Sound of AI" youtube tutorial. 


The dataset used is the Google "Speech Commands" Dataset, and can be found [here](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 

As this project is more about learning AWS and Docker, and I am less interested in the deep learning model portion, I cut down the training to only 8 commands (the full dataset has 10). 

## Preprocessing

The "prepare_dataset.py" file walks through the entire dataset folder ands creates a large json file with the command name ("up", "down", etc), the training label (i.e an integer representation of the command name - just the folder number in this case), the filename, and the mel-frequency cepstrum coefficients (MFCCS). Details on what MFCCs are can be found [here](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd) and [here](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), they are widely used in speech recognition. 


## Training 

A convolutional neural net (CNN) is trained by first splitting the dataset into training, test, and validation sets. The model was trained for 40 epochs and achieved an accuracy of 0.94 - this could certainly be improved but was deemed sufficient for this usage. 

The model was saved as a model.h5 file for future predictions. 

The model architecutre is shown below: 

![model_figure](https://user-images.githubusercontent.com/66977019/120940143-a0e82e00-c6d0-11eb-8994-adf87c96cbef.png)

