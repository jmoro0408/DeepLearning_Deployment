import librosa
import os
import json


DATASET_PATH = r"dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 #1 second worth of sound using librosa default sampling

def prepare_dataset(dataset_path, json_path, n_mfcc = 13, hop_length = 512, n_fft = 2048):

    #data dictionary
    data = {
        "mappings" : [], 
        "labels" : [], 
        "MFCCs": [], 
        "files":[]
            }


    # loop through all the sub dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # need to make sure we're not at root  level
        if dirpath is not dataset_path:
            #update mappings
            category = dirpath.split("/")[-1] # "dataset/dowm" -? [dataset, down] 
            data["mappings"].append(category)
            print(f"Processing {category}")

            #loop through and extract MFCCs
            for f in filenames:
                #get file path
                file_path = os.path.join(dirpath, f)
                #load audio file
                signal, sr = librosa.load(file_path)

                #ensure audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    #enforce 1 second long signal
                    signal = signal[:SAMPLES_TO_CONSIDER] # trimming >1 s long data to 1 sec

                    #extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, hop_length = hop_length, n_fft = n_fft)
                    #store data

                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist()) #convert np array to list for json storage
                    data["files"].append(file_path)
                    print(f"{file_path} : {i-1}")

    #store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)

        
if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)









