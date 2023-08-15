import tensorflow as tf

mnist = tf.keras.datasets.mnist
#load dataset
(x_train, y_train), (x_test, y_test) =  mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
#print(predictions)

#The tf.nn.softmax function converts these logits to probabilities for each class
predictions = tf.nn.softmax(predictions).numpy()
#print(predictions)

#Define a loss fn for training using 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = loss_fn(y_train[:1], predictions).numpy()
#print(loss_fn)

#Configure and compile the model
model.compile('adam', loss=None,metrics=['accuracy'])

#Train and evaluate your model - Use the Model.fit method to adjust your model parameters and minimize the loss
#print(model.fit(x_train, y_train, epochs=5))
model.evaluate(x_test, y_test, verbose=2)
probability_model=tf.keras.Sequential([model, tf.keras.layers.Softmax()])
probability_model(x_test[:5])
