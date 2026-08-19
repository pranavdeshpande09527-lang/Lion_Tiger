import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ["lion", "tiger"]

# Load the model from the first training stage.
model = tf.keras.models.load_model("lion_tiger_classifier.keras")

# Find the pretrained EfficientNet part of the model.
base_model = model.get_layer("efficientnetb0")

# Allow the last part of EfficientNet to learn from lion/tiger images.
base_model.trainable = True

# Keep most layers frozen; fine-tune only the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Load the original datasets.
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    class_names=CLASS_NAMES,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    "data/validation",
    class_names=CLASS_NAMES,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    class_names=CLASS_NAMES,
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Use a very small learning rate for careful adjustments.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Stop if validation loss does not improve for 3 epochs.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=8,
    callbacks=[early_stopping],
)

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nFine-tuned test accuracy: {test_accuracy:.2%}")

model.save("lion_tiger_classifier_finetuned.keras")
print("Saved: lion_tiger_classifier_finetuned.keras")