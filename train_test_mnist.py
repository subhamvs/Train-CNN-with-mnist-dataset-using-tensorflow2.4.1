import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.preprocessing.image as image
import numpy as np
import sys

# Develop the model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(), 
    keras.layers.Dense(800, activation='relu'), 
    keras.layers.Dense(10, activation='sigmoid')  
])

#  Compile the model
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

# Print the summary of the model
print (model.summary())

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 16 using train_datagen generator
# --------------------
train_dir = "<Path of mnist path>\\mnist_png\\training\\"
test_dir = "<Path of mnist path>\\mnist_png\\testing\\"
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    target_size=(28, 28),
                                                    color_mode = 'grayscale')     
# --------------------
# Flow validation images in batches of 16 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(test_dir,
                                                         batch_size=16,
                                                         class_mode  = 'categorical',
                                                         target_size = (28, 28),
                                                         color_mode = 'grayscale')

# Train the model
print ("\nTrain the model")
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_steps=50,
                    verbose=2)

# Predict the images
image_name_list = ["1\\2.png", 
                  "2\\1.png", 
                  "3\\18.png", 
                  "4\\4.png", 
                  "5\\8.png", 
                  "6\\11.png", 
                  "7\\0.png", 
                  "8\\61.png", 
                  "9\\7.png", 
                  "0\\3.png"]

# Test the images
print ("\nTest the sample images")
for image_name in image_name_list:
    print (image_name)
    img=image.load_img(test_dir+image_name, target_size=(28, 28), color_mode = "grayscale")
    
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)

    classes = model.predict(x)
    print ("Class_id predicted : " + str(np.argmax(classes)))

    