from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_hate(text, model, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict(padded)
    print(prediction)
    return prediction.argmax(axis=-1)[0]
