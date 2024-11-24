from datasets import load_dataset
import numpy as np

class dataset_:
    

    def __init__(self):
        
        train_ = load_dataset("badmatr11x/hate-offensive-speech", split="train")
        val_ = load_dataset("badmatr11x/hate-offensive-speech", split="validation")
        test_ = load_dataset("badmatr11x/hate-offensive-speech", split="test")
        
        self.train_texts = [item["tweet"] for item in train_]
        self.train_labels = np.array([item["label"] for item in train_])
        
        self.val_texts = [item["tweet"] for item in val_]
        self.val_labels = np.array([item["label"] for item in val_])
        
        self.test_texts = [item["tweet"] for item in test_]
        self.test_labels =np.array([item["label"] for item in test_])

