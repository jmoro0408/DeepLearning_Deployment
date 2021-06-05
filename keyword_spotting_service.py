import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
from data_preprocess import (
    SAMPLES_TO_CONSIDER,
)  # 1 second worth of sound with our sample rate


class _Keyword_Spotting_Service:

    model = None
    _mappings = ["right", "no", "left", "up", "down", "yes", "on", "off"]

    _instance = None

    def predict(self, file_path):
        # extract the MFCCs
        MFCCs = self.preprocess(file_path)  # Shape = (#segments, #coefficients)
        # Convert 2D MFCCs arrray into 4D array ->(#samples, #segments, #coefficients, #channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)  # find highest prediciton
        predicted_keyword = self._mappings[predicted_index]  # lookup index in mappings
        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)
        # ensure consistency in the aidio file length
        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]
        # extract MFCCs
        MFCCs = librosa.feature.mfcc(
            signal,sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        return MFCCs.T


def Keyword_Spotting_Service():

    # Ensure we only have 1 instance of KSS

    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict("test/left.wav")
    keyword2 = kss.predict("test/left.wav")

    print(f"Predicted keywords: {keyword1}, {keyword2}")
