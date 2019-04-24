import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

class handwrittenCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = handwrittenCallBack()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0])
print(y_train[0])

x_train = x_train/255
x_test = x_test/255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, callbacks=[callbacks])

model.evaluate(x_test, y_test)

classifications = model.predict(x_test)
print(classifications[0])