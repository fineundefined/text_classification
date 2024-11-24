import model_create
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_model(vocab_size,max_length, model_name,dataset):
    tokenizer = Tokenizer(num_words=vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(dataset.train_texts)

    train_sequences = tokenizer.texts_to_sequences(dataset.train_texts)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post")

    val_sequences = tokenizer.texts_to_sequences(dataset.val_texts)
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post")

    test_sequences = tokenizer.texts_to_sequences(dataset.test_texts)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post")
    
    model = model_create.hate_class(vocab_size,max_length)


    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(
        train_padded,
        dataset.train_labels,
        epochs=10,
        validation_data=(val_padded,dataset.val_labels),
        verbose=1
    )
    accuracy = model.evaluate(test_padded, dataset.test_labels)
    print(f"Test dataset accuracy: {accuracy:.2f}")
    
    model.save(model_name)
    
    