import tensorflow.keras as keras
MODEL_PATH = "model.h5"


class Keyword_Spotting_Service:

    model = None
    _mappings = [
        "right",
        "no",
        "left",
        "up",
        "down",
        "yes",
        "on",
        "off"
    ]

    _instance = None

    def predict(self, file_path):
        #extract the MFCCs
        MFCCS = self.preprocess(file_path) #Shape = (#segments, #coefficients)
        #Convert 2D MFCCs arrray into 4D array ->(#samples, #segments, #coefficients, #channels)
        


        #make prediction


        def preprocess(self, file_path):
            pass


def Keyword_Spotting_Service():

    #Ensure we only have 1 instance of KSS

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

