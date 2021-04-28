#using artificial neural network

#from tensorflow.keras.models import Sequential 
#from keras.layers import Dense

ann_model=Sequential()

ann_model.add( Dense(8, activation="relu", input_shape = (x.shape[1],) ) )
ann_model.add( Dense(8, activation="relu") )
ann_model.add( Dense(1, activation="sigmoid"))

ann_model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=['accuracy'])

#validation set

x_val=x[:1000]
y_val=y[:1000]

x_val.shape

y_val.shape

result = ann_model.fit(x, y, batch_size=10, epochs=100, validation_data=(x_val, y_val),verbose=1)

ann_model.evaluate(x_val, y_val)
