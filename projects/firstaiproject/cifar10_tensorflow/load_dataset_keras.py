import tensorflow as tf

#load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
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
print(loss_fn)