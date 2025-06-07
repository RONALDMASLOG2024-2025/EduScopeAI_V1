from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json

# Load data
img_size = (224, 224)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=img_size,
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=img_size,
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Use InceptionV3 as base model
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(*img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save the model
model.save('model/microorganism_model.keras')
print("✅ Model trained and saved using Kaggle data!")



# Save class labels
class_labels = list(train_data.class_indices.keys())
with open("model/class_names.json", "w") as f:
    json.dump(class_labels, f)
