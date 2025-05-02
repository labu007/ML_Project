import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Parameters
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
EPOCHS = 25

train_dir = 'split/train'
test_dir = 'split/test'

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=2)
test_ds = test_ds.prefetch(buffer_size=2)

# Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model_dr_ratino_model.h5', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Train model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

import json

# Save the training history as a JSON file
def save_training_history(hist, filename="training_history.json"):
    with open(filename, 'w') as f:
        json.dump(hist.history, f)
    print(f"Training history saved to {filename}")

# Save the history after training
save_training_history(history)


###test 
# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy and loss plot
def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Test Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Testing Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    
    plt.show()

plot_history(history)

# Generate a table of training and testing accuracy/loss
def display_metrics_table(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    # Print table header
    print(f"{'Epoch':<6}{'Training Accuracy':<20}{'Testing Accuracy':<20}{'Training Loss':<20}{'Testing Loss':<20}")
    print("-" * 80)

    # Print metrics for each epoch
    for epoch, train_acc, test_acc, train_loss, test_loss in zip(epochs_range, acc, val_acc, loss, val_loss):
        print(f"{epoch:<6}{train_acc:<20.4f}{test_acc:<20.4f}{train_loss:<20.4f}{test_loss:<20.4f}")

# Display the metrics table
display_metrics_table(history)


# Grad-CAM Visualization


import matplotlib.cm as cm
from PIL import Image


# Function to preprocess a single image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Grad-CAM heatmap generation function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Overlay heatmap on image
def overlay_heatmap(heatmap, img_path, alpha=0.4):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize(img.size)
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = np.array(img) * (1 - alpha) + jet_heatmap * alpha
    superimposed_img = np.uint8(superimposed_img)
    return Image.fromarray(superimposed_img)

# Example usage
img_path = 'split/test/No_DR/98441214557f.png'  # Replace with your image path
img_array = preprocess_image(img_path, IMG_SIZE)

# Use the EfficientNetB0's last convolutional layer
last_conv_layer_name = 'top_conv'  # Modify based on your model structure

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
superimposed_img = overlay_heatmap(heatmap, img_path)

# Save and display the result
superimposed_img.save('grad_cam_output.jpg')
superimposed_img.show()