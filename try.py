from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import data_process
import predict

##################
from model_ import create_model

if __name__ == "__main__":
    # create dataset instance
    dataset = data_process.dataset_()
    
    
    vocab_size = 10000
    max_length = 100
    model_name = "hate_class.h5"
    
    create_model(vocab_size,max_length,model_name,dataset)
    
    model = load_model(model_name)
    
    # create tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(dataset.train_texts)


    predicts = {0: "hate speech", 1: "offensive speech", 2: "normal speech"}

    while True:
        user_input = input("Type text to classify: ")
        answer = predict.predict_hate(user_input, model, tokenizer, max_length)
        print(f"Your message contains {predicts[answer]}")
        
        