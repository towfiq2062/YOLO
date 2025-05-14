import os
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directories for data
train_dirs = {
    'T1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/TSL/train',
    'T2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/ECSL/train',
    'T3': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/ESCSL/train',
    'HL1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/heat_lamp/1/train',
    'HL2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/heat_lamp/2/train',
    'data': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/DATA/train/train'
}


val_dirs = {
    'T1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/TSL/val',
    'T2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/ECSL/val',
    'T3': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/crate_type/ESCSL/val',
    'HL1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/heat_lamp/1/val',
    'HL2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/train/heat_lamp/2/val',
    'data': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/DATA/train/val'
}



test_dirs = {
    'T1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/test1/crate_type/TSL',
    'T2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/test1/crate_type/ECSL',
    'T3': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/test1/crate_type/ESCSL',
    'HL1': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/test1/heat_lamp/1',
    'HL2': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/BLOCK/test1/heat_lamp/1',
    'data': '/mnt/nrdstor/amilab/towfiqami/Final/DATA/data12-15-2024/DATA/test1'
}

output_dir = '/mnt/nrdstor/amilab/towfiqami/Final/codes/RESNET/All-train-test'

# Training loop
for dataset_name in train_dirs.keys():
    print(f"Training on dataset: {dataset_name}")

    train_dir = train_dirs[dataset_name]
    val_dir = val_dirs[dataset_name]
    test_dir = test_dirs[dataset_name]

    # Data generators
    datagen_trainval = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = datagen_trainval.flow_from_directory(
        train_dir, target_size=(220, 220), batch_size=64, class_mode='categorical', shuffle=True)
    valid_generator = datagen_trainval.flow_from_directory(
        val_dir, target_size=(220, 220), batch_size=64, class_mode='categorical', shuffle=True)
    test_generator = datagen_trainval.flow_from_directory(
        test_dir, target_size=(220, 220), batch_size=64, class_mode='categorical', shuffle=False)

    # Build ResNet50 model
    resnet_model = Sequential([
        tf.keras.applications.ResNet50(include_top=False, input_shape=(220, 220, 3), pooling='avg', weights='imagenet'),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.05)),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compile model
    resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    model_checkpoint_path = os.path.join(output_dir, f"{dataset_name}_best_model.h5")
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = resnet_model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=50,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save model
    model_save_path = os.path.join(output_dir, f"{dataset_name}_final_model.h5")
    resnet_model.save(model_save_path)

    # Evaluate on test data
    test_loss, test_accuracy = resnet_model.evaluate(test_generator)
    print(f"Test Accuracy for {dataset_name}: {test_accuracy}")

    # Confusion matrix
    predictions = resnet_model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    cm = confusion_matrix(true_classes, predicted_classes)
    cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    report_path = os.path.join(output_dir, f"{dataset_name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Saved results for {dataset_name} in {output_dir}")
