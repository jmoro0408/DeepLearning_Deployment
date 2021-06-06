# Deep Learning Deployment with Docker and AWS

Small project to increase my knowledge of Docker and AWS deployment. 
The project uses a simple convolutional neural network (CNN) to make predictions on some simple voice commands ("up", "down", "right", "left" and others). The code follows a "The Sound of AI" youtube tutorial. 


The dataset used is the Google "Speech Commands" Dataset, and can be found [here](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 


## Preprocessing

The "prepare_dataset.py" file walks through the entire dataset folder ands creates a large json file with the command name ("up", "down", etc), the training label (i.e an integer representation of the command name - just the folder number in this case), the filename, and the mel-frequency cepstrum coefficients (MFCCS). Details on what MFCCs are can be found [here](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd) and [here](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), they are widely used in speech recognition. 

An example of an MFCC for a 1 second voice clip is shown below. 

<img src="https://user-images.githubusercontent.com/66977019/120943171-1c061000-c6e2-11eb-8060-b01df42367e7.png" width = "550">

## Training 

A convolutional neural net (CNN) is trained by first splitting the dataset into training, test, and validation sets. The model was trained for 40 epochs and achieved an accuracy of 0.94 - this could certainly be improved but was deemed sufficient for this usage. 

The model was saved as a model.h5 file for future predictions. 

The model architecture is shown below: 

![model_figure](https://user-images.githubusercontent.com/66977019/120940143-a0e82e00-c6d0-11eb-8994-adf87c96cbef.png)

## Making Predictions

First I tested the predictions locally, using model.predict with some test audio files.  A client.py file simulates a client post request to a simple flask development server (server.py) that accepts temporarily stores the passed audio file, makes a prediction with the trained model, and pings back the predicted keyword back to the client before removing the audio file. 

## NGINX and uWSGI

A uWSGI server is placed between the client and the flask development server to implement the WSGI spec required by nginx. The flask app is then called by uwsgi when required.  A nice blog post detailing why we need uWSGI can be found [here](https://www.ultravioletsoftware.com/single-post/2017/03/23/An-introduction-into-the-WSGI-ecosystem). 


NGINX is then placed in front of uWSGI, directly infront of the client. 
The client fires a http post request with an audio file, nginx picks up the request and proxies it to uWSGI to call the flask app and make the prediction. 

NGINX is then wrapped in its own Docker container, with uWSGI, flask and the Tensorflow model in a separate one. 

## Deployment on AWS

Finally all server side files are transferred to a standard Amazon web services EC2 instance and the docker. 

# Conclusion

This project was great as it had a greater focus on production deployment rather than on the deep learning aspect. Using Docker, Flask, AWS, and NGINX/uWSGI were all new aspects to me which I strive to include in future projects. 
