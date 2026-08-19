import tensorflow as tf
import json

# Load trained model
model = tf.keras.models.load_model("lion_tiger_classifier.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load data2
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data2",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Test model
loss, accuracy = model.evaluate(test_ds)

print(f"\nData2 Test Accuracy: {accuracy * 100:.2f}%")
print(f"Data2 Test Loss: {loss:.4f}")
print("Classes:", class_names)