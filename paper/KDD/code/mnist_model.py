model = Sequential()
model.add(Dense(10, input_shape=(784,), 
          kernel_initializer='zeros', 
          bias_initializer='zeros'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              metrics=['accuracy'])

history = model.fit(X_train, 
                    Y_train, 
                    epochs=500, 
                    batch_size=128, 
                    validation_data=(X_test, Y_test), 
                    verbose=1)