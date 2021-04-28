from tensorflow .keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM,Embedding

rnn_model=Sequential()

#validation set

x_val=x[:1000]
y_val=y[:1000]

x.shape

rnn_model.add( Embedding(200, 32, input_length=50) ) 
rnn_model.add( LSTM(50) )
rnn_model.add( Dense(1, activation="sigmoid")) 
rnn_model.add( Dense(1, activation="sigmoid"))

rnn_model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['accuracy'])

rnn_model.fit(x, y, batch_size=100, epochs=5, validation_data=(x_val, y_val),verbose=1)
rnn_model.evaluate(x_val,y_val)
