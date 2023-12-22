# 3層モデル提出用
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution(x_test_df, train_df):
    train_df = train_df[['waferMap', 'failureType']]
    x_test_df = x_test_df[['waferMap']]

    # データ拡張ジェネレータの設定
    data_gen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True)


    def preprocess_images_tf(image_list, size=(32, 32)):
        processed_images = []
        for img in image_list:
            img = np.where(img == 2, 1, 0)
            img = np.expand_dims(img, axis=-1)
            img = tf.image.resize(img, size)
            processed_images.append(img)
        return np.array(processed_images)

    failure_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df['failureType'])

    train_images = preprocess_images_tf(train_df['waferMap'].tolist())
    test_images = preprocess_images_tf(x_test_df['waferMap'].tolist())

    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(len(failure_types), activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # データ拡張を適用してトレーニングデータを準備
    augmented_images = [data_gen.random_transform(img) for img in train_images]
    augmented_images = np.array(augmented_images)

    model.fit(train_images, train_labels_encoded, batch_size=64, epochs=40)

    predictions = model.predict(test_images)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    return pd.DataFrame({'failureType': predicted_labels}, index=x_test_df.index)
