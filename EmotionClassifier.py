import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
from pathlib import Path
import seaborn as sns
from collections import Counter
import kagglehub
import json
import pandas as pd
from datetime import datetime

# Download dataset
try:
    path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    print(f"Dataset downloaded to: {path}")
except Exception as e:
    print(f"KaggleHub download failed: {e}")
    path = None


class EmotionClassifier:
    def __init__(self):
        self.image_size = 48
        self.emotions = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
        
        # Initialize face detection
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

        self.model = None
        self.history = None

    def load_rafdb_data(self, path, max_per_class=None):
        """Load RAF-DB data from the correct structure"""
        X = []
        y = []
        
        print(f"Looking for dataset at: {path}")

        if not path or not os.path.exists(path):
            print(f"Dataset path does not exist: {path}")
            raise FileNotFoundError(f"RAF-DB dataset not found at {path}")

        # RAF-DB numeric to emotion mapping
        emotion_mapping = {
            '1': 5,  # Angry
            '2': 3,  # Happy
            '3': 4,  # Sad
            '4': 2,  # Disgust
            '5': 1,  # Fear
            '6': 0,  # Surprise
            '7': 6   # Neutral
        }

        dataset_paths = [
            os.path.join(path, "DATASET", "train"),
            os.path.join(path, "DATASET", "test"),
            os.path.join(path, "train"),
            os.path.join(path, "test")
        ]

        total_loaded = 0
        
        for base_path in dataset_paths:
            if not os.path.exists(base_path):
                continue
                
            print(f"Loading from: {base_path}")
            
            for emotion_num, emotion_idx in emotion_mapping.items():
                emotion_folder = os.path.join(base_path, emotion_num)
                
                if not os.path.exists(emotion_folder):
                    print(f"  Folder {emotion_num} not found in {base_path}")
                    continue
                
                count = 0
                for fname in os.listdir(emotion_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(emotion_folder, fname)
                        try:
                            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                            if img is None:
                                continue
                            img = cv2.resize(img, (self.image_size, self.image_size))
                            X.append(img)
                            y.append(emotion_idx)
                            count += 1
                            total_loaded += 1
                            
                            if max_per_class and count >= max_per_class:
                                break
                                
                        except Exception as e:
                            continue
                
                emotion_name = self.emotions[emotion_idx]
                print(f"  Loaded {count} images for {emotion_name} (folder {emotion_num})")

        if len(X) == 0:
            print("ERROR: No images loaded from dataset!")
            raise RuntimeError("Could not load any images from dataset")

        print(f"\nSuccessfully loaded {len(X)} images")
        return X, y

    def analyze_data_distributions(self, labels):
        """Analyze and visualize class distribution"""
        label_counts = Counter(labels)

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        emotion_names = [self.emotions[i] for i in label_counts.keys()]
        counts = list(label_counts.values())
        
        bars = plt.bar(emotion_names, counts)
        plt.title('Emotion Class Distribution')
        plt.xlabel('Emotions')
        plt.ylabel('Number Of Images')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=emotion_names, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (%)')
        
        plt.tight_layout()
        plt.show()

        print("\nClass Distribution:")
        for emotion_idx, count in label_counts.items():
            print(f"  {self.emotions[emotion_idx]}: {count} samples")

    def prepare_data(self, X, y, use_3_channels=False):
        """Prepare and normalize data"""
        X_arr = np.array(X, dtype='float32')
        if X_arr.ndim == 3:
            if use_3_channels:
                # Convert to 3 channels for transfer learning
                X_arr = np.stack([X_arr] * 3, axis=-1)
            else:
                X_arr = np.expand_dims(X_arr, axis=-1)
        X_arr /= 255.0

        y_arr = np.array(y, dtype='int32')
        y_cat = keras.utils.to_categorical(y_arr, num_classes=len(self.emotions))
        
        return X_arr, y_cat

    def build_cnn_model(self):
        """Build a CNN model for grayscale images"""
        model = keras.Sequential([
            layers.Input(shape=(self.image_size, self.image_size, 1)),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth conv block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(len(self.emotions), activation='softmax')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("CNN Model Created!")
        print(f"Total parameters: {model.count_params():,}")
        model.summary()
        return model

    def build_transfer_learning_model(self):
        """Build model using transfer learning with MobileNetV2"""
        try:
            # Use MobileNetV2 as base (lighter than EfficientNet)
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.image_size, self.image_size, 3)
            )
            base_model.trainable = False  # Freeze base model
            
            model = keras.Sequential([
                layers.Input(shape=(self.image_size, self.image_size, 3)),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(len(self.emotions), activation='softmax')
            ])
            
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            print("Transfer Learning Model Created!")
            print(f"Total parameters: {model.count_params():,}")
            model.summary()
            return model
            
        except Exception as e:
            print(f"Transfer learning failed: {e}")
            print("Falling back to CNN model...")
            return self.build_cnn_model()

    def train(self, X_train, y_train, X_val, y_val, use_augmentation=True, epochs=50, batch_size=32):
        """Train the model"""
        if self.model is None:
            raise RuntimeError('Model not built. Call build_model() first.')

        # Compute class weights
        class_weight = None
        try:
            if y_train.ndim == 2:
                y_labels = np.argmax(y_train, axis=1)
            else:
                y_labels = y_train
            classes = np.unique(y_labels)
            weights = compute_class_weight('balanced', classes=classes, y=y_labels)
            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
            print(f"Class weights: {class_weight}")
        except Exception as e:
            print(f"Could not compute class weights: {e}")
            class_weight = None

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]

        if use_augmentation:
            print("Using data augmentation...")
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            )
            
            hist = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
        else:
            hist = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
        
        self.history = hist
        return hist

    def evaluate(self, X_val, y_val):
        """Evaluate the model"""
        if self.model is None:
            raise RuntimeError('Model not built. Call build_model() first.')
        
        return self.model.evaluate(X_val, y_val, verbose=0)

    def webcam_demo(self):
        """Webcam emotion detection demo"""
        print("\nStarting WebCam - press 'q' to quit")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        emotion_colors = {
            'Surprise': (255, 255, 0),    # Cyan
            'Fear': (255, 0, 255),        # Magenta  
            'Disgust': (0, 255, 0),       # Green
            'Happy': (0, 255, 255),       # Yellow
            'Sad': (255, 0, 0),           # Blue
            'Angry': (0, 0, 255),         # Red
            'Neutral': (255, 255, 255),   # White
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Extract face
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (self.image_size, self.image_size))
                    face_img = face_img.astype('float32') / 255.0
                    
                    # Prepare input based on model type
                    if self.model.input_shape[-1] == 1:
                        # CNN model - use grayscale
                        face_img = np.expand_dims(face_img, axis=(0, -1))
                    else:
                        # Transfer learning model - convert to 3 channels
                        face_img = np.stack([face_img] * 3, axis=-1)
                        face_img = np.expand_dims(face_img, axis=0)
                    
                    # Predict emotion
                    predictions = self.model.predict(face_img, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    confidence = predictions[0][emotion_idx]
                    emotion = self.emotions[emotion_idx]
                    
                    # Draw rectangle and label
                    color = emotion_colors.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def the_main():
    classifier = EmotionClassifier()

    print("="*60)
    print("STEP 1: Building model...")
    print("="*60)
    # Use CNN model to avoid transfer learning issues
    print("Using CNN model for better compatibility...")
    classifier.build_cnn_model()

    print("\n" + "="*60)
    print("STEP 2: Loading data...")
    print("="*60)
    try:
        X, y = classifier.load_rafdb_data(path)
        
        print(f"Loaded {len(X)} images")
        if len(X) > 0:
            print(f"Image shape: {X[0].shape}")
            print(f"Image dtype: {X[0].dtype}")
            print(f"Image value range: [{X[0].min()}, {X[0].max()}]")
        
        classifier.analyze_data_distributions(y)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n" + "="*60)
    print("STEP 3: Preparing data...")
    print("="*60)
    # Use grayscale for CNN model
    x_processed, y_processed = classifier.prepare_data(X, y, use_3_channels=False)
    print(f"Processed data shape: {x_processed.shape}")
    print(f"Processed data range: [{x_processed.min():.3f}, {x_processed.max():.3f}]")

    print("\n" + "="*60)
    print("STEP 4: Splitting data...")
    print("="*60)
    X_train, X_val, y_train, y_val = train_test_split(
        x_processed, y_processed, 
        test_size=0.2, 
        stratify=np.argmax(y_processed, axis=1),
        random_state=42
    )
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    print("\n" + "="*60)
    print("STEP 5: Training model...")
    print("="*60)
    # Use smaller batch size for stability
    history = classifier.train(
        X_train, y_train, X_val, y_val, 
        use_augmentation=True, 
        epochs=50,
        batch_size=32
    )

    print("\n" + "="*60)
    print("STEP 6: Evaluating model...")
    print("="*60)
    loss, accuracy = classifier.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    print("\n" + "="*60)
    print("STEP 7: Starting webcam demo...")
    print("="*60)
    try:
        classifier.webcam_demo()
    except Exception as e:
        print(f"Webcam demo failed: {e}")


if __name__ == "__main__":
    the_main()