from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import cv2


"""1. Visualization Utilities

1.1 Bounding Boxes Utilities
We have provided you with some functions which you will use to draw bounding boxes around the birds in the image.

draw_bounding_box_on_image: Draws a single bounding box on an image.
draw_bounding_boxes_on_image: Draws multiple bounding boxes on an image.
"""


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5):
    """
    Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
    """

    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)


def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=5):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                   boxes[i, 2], color[i], thickness)
    return image


"""1.2 Data and Predictions Utilities
We've given you some helper functions and code that are used to visualize the data and the model's predictions.

display_digits_with_boxes: This displays a row of "digit" images along with the model's predictions for each image.
plot_metrics: This plots a given metric (like loss) as it changes over multiple epochs of training.
"""

# Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')  # Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


# utility to display a row of digits with their predictions
def display_digits_with_boxes(images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False):
    n = len(images)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if (len(pred_bboxes) > i):
            bbox = pred_bboxes[i]
            bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                    bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        if (len(bboxes) > i):
            bbox = bboxes[i]
            if bboxes_normalized == True:
                bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                        bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        img_to_draw = draw_bounding_boxes_on_image(image=images[i], boxes=np.asarray(bboxes_to_plot),
                                                   color=[(255, 0, 0), (0, 255, 0)])
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if (iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (iou[i][0]), color=color, transform=ax.transAxes)


# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)


"""2. Preprocess and Load the Dataset

2.1 Preprocessing Utilities
We have given you some helper functions to pre-process the image data.

read_image_tfds
Resizes image to (224, 224)
Normalizes image
Translates and normalizes bounding boxes
"""


def read_image_tfds(image, bbox):
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(image, (224, 224,))

    image = image / 127.5
    image -= 1

    bbox_list = [bbox[0] / factor_x,
                 bbox[1] / factor_y,
                 bbox[2] / factor_x,
                 bbox[3] / factor_y]

    return image, bbox_list


"""read_image_with_shape
This is very similar to read_image_tfds except it also keeps a copy of the original image (before pre-processing) and returns this as well.

Makes a copy of the original image.
Resizes image to (224, 224)
Normalizes image
Translates and normalizes bounding boxes
"""


def read_image_with_shape(image, bbox):
    original_image = image

    image, bbox_list = read_image_tfds(image, bbox)

    return original_image, image, bbox_list



"""read_image_tfds_with_original_bbox
This function reads image from data
It also denormalizes the bounding boxes 
(it undoes the bounding box normalization that is performed by the previous two helper functions.)
"""

def read_image_tfds_with_original_bbox(data):
    image = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)
    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    bbox_list = [bbox[1] * factor_x ,
                 bbox[0] * factor_y,
                 bbox[3] * factor_x,
                 bbox[2] * factor_y]
    return image, bbox_list


"""dataset_to_numpy_util
This function converts a dataset into numpy arrays of images and boxes.

This will be used when visualizing the images and their bounding boxes
"""


def dataset_to_numpy_util(dataset, N=0):
    # eager execution: loop through datasets normally
    take_dataset = dataset.shuffle(1024)

    if N > 0:
        take_dataset = take_dataset.take(N)

    ds_images, ds_bboxes = [], []
    for images, bboxes in take_dataset:
        ds_images.append(images.numpy())
        ds_bboxes.append(bboxes.numpy())

    return (np.array(ds_images), np.array(ds_bboxes))



"""dataset_to_numpy_with_original_bboxes_util
This function converts a dataset into numpy arrays of
original images
resized and normalized images
bounding boxes
This will be used for plotting the original images with true and predicted bounding boxes.
"""


def dataset_to_numpy_with_original_bboxes_util(dataset, N=0):
    normalized_dataset = dataset.map(read_image_with_shape)

    if N > 0:
        normalized_dataset = normalized_dataset.take(N)

    ds_original_images, ds_images, ds_bboxes = [], [], []

    for original_images, images, bboxes in normalized_dataset:
        ds_images.append(images.numpy())
        ds_bboxes.append(bboxes.numpy())
        ds_original_images.append(original_images.numpy())

    return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)



"""2.2 Visualize the images and their bounding box labels
Now you'll take a random sample of images from the training and validation sets and visualize them by plotting the corresponding bounding boxes.

Visualize the training images and their bounding box labels
"""


import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def get_visualization_training_dataset():
    dataset, info = tfds.load("caltech_birds2010", split="train", with_info=True, data_dir=data_dir, download=False)
    print(info)
    visualization_training_dataset = dataset.map(read_image_tfds_with_original_bbox,
                                                 num_parallel_calls=16)
    return visualization_training_dataset


visualization_training_dataset = get_visualization_training_dataset()


(visualization_training_images, visualization_training_bboxes) = dataset_to_numpy_util(visualization_training_dataset, N=10)
display_digits_with_boxes(np.array(visualization_training_images), np.array([]), np.array(visualization_training_bboxes), np.array([]), "training images and their bboxes")



"""Visualize the validation images and their bounding boxes"""

def get_visualization_validation_dataset():
    dataset = tfds.load("caltech_birds2010", split="test", data_dir=data_dir, download=False)
    visualization_validation_dataset = dataset.map(read_image_tfds_with_original_bbox, num_parallel_calls=16)
    return visualization_validation_dataset


visualization_validation_dataset = get_visualization_validation_dataset()

(visualization_validation_images, visualization_validation_bboxes) = dataset_to_numpy_util(visualization_validation_dataset, N=10)
display_digits_with_boxes(np.array(visualization_validation_images), np.array([]), np.array(visualization_validation_bboxes), np.array([]), "validation images and their bboxes")


"""2.3 Load and prepare the datasets for the model
These next two functions read and prepare the datasets that you'll feed to the model.

They use read_image_tfds to resize, and normalize each image and its bounding box label.
They performs shuffling and batching.
You'll use these functions to create training_dataset and validation_dataset, which you will give to the model that you're about to build.
"""

BATCH_SIZE = 64

def get_training_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(-1)
    return dataset

def get_validation_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset

training_dataset = get_training_dataset(visualization_training_dataset)
validation_dataset = get_validation_dataset(visualization_validation_dataset)


"""3. Define the Model
Bounding box prediction is treated as a "regression" task, in that you want the model to output numerical values.

You will be performing transfer learning with MobileNet V2. The model architecture is available in TensorFlow Keras.
You'll also use pretrained 'imagenet' weights as a starting point for further training. These weights are also readily available
You will choose to retrain all layers of MobileNet V2 along with the final classification layers.
Note: For the following exercises, please use the TensorFlow Keras Functional API (as opposed to the Sequential API).
"""


def feature_extractor(inputs):
    # Create a mobilenet version 2 model object
    mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                                                     include_top=False,
                                                                     weights='imagenet')

    # pass the inputs into this modle object to get a feature extractor for these inputs
    feature_extractor = mobilenet_model(inputs)

    # return the feature_extractor
    return feature_extractor


def dense_layers(features):
    # global average pooling 2D layer.
    x = tf.keras.layers.GlobalAveragePooling2D()(features)

    # 1024 Dense layer, with relu
    x = tf.keras.layers.Dense(1024, activation="relu")(x)

    # 512 Dense layer, with relu
    x = tf.keras.layers.Dense(512, activation="relu")(x)

    return x

def bounding_box_regression(x):

    # Dense layer named `bounding_box`
    bounding_box_regression_output = tf.keras.layers.Dense(4 , name='bounding_box')(x)

    return bounding_box_regression_output


def final_model(inputs):
    # features
    feature_cnn = feature_extractor(inputs)

    # dense layers
    last_dense_layer = dense_layers(feature_cnn)

    # bounding box
    bounding_box_output = bounding_box_regression(last_dense_layer)

    # define the TensorFlow Keras model using the inputs and outputs to your model
    model = tf.keras.models.Model(inputs=inputs, outputs=bounding_box_output)

    return model


def define_and_compile_model():
    # define the input layer
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    # create the model
    model = final_model(inputs)

    # compile your model
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss='mse')

    return model


"""Run the cell below to define your model and print the model summary."""

# define your model
model = define_and_compile_model()

# print model layers
tf.keras.utils.plot_model(model,show_shapes=True)


"""Train the Model"""

import math
EPOCHS = 50

# Choose a batch size
BATCH_SIZE = 32

# Get the length of the training set
length_of_training_dataset = len(visualization_training_dataset)

# Get the length of the validation set
length_of_validation_dataset = len(visualization_validation_dataset)

# Get the steps per epoch (
steps_per_epoch = math.ceil(length_of_training_dataset / BATCH_SIZE)

# get the validation steps (per epoch)
validation_steps = math.ceil(length_of_validation_dataset / BATCH_SIZE)

history = model.fit(training_dataset , steps_per_epoch = steps_per_epoch ,
                    validation_data = validation_dataset , validation_steps=validation_steps , epochs=EPOCHS)


"""5. Validate the Model

5.1 Loss
You can now evaluate your trained model's performance by checking its loss value on the validation set.
"""


loss = model.evaluate(validation_dataset, steps=validation_steps)
print("Loss: ", loss)


""".3 Plot Loss Function
You can also plot the loss metrics.
"""

plot_metrics("loss", "Bounding Box Loss", ylim=0.2)

"""5.4 Evaluate performance using IoU
You can see how well your model predicts bounding boxes on the validation set by calculating the Intersection-over-union (IoU) score for each image.

You'll find the IoU calculation implemented for you.
Predict on the validation set of images.
Apply the intersection_over_union on these predicted bounding boxes.
"""

def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  np.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)

    #Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    #Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    #Calculates overlap area and union area.
    overlap_area = np.maximum((xmax_overlap - xmin_overlap),0)  * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    # Defines a smoothing factor to prevent division by 0
    smoothing_factor = 1e-10

    #Updates iou score
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou

#Makes predictions
original_images, normalized_images, normalized_bboxes = dataset_to_numpy_with_original_bboxes_util(visualization_validation_dataset, N=500)
predicted_bboxes = model.predict(normalized_images, batch_size=32)

#Calculates IOU and reports true positives and false positives based on IOU threshold
iou = intersection_over_union(predicted_bboxes, normalized_bboxes)
iou_threshold = 0.5

print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))


"""Visualize Predictions
Lastly, you'll plot the predicted and ground truth bounding boxes for a random set of images and visually see how well you did!
"""

n = 10
indexes = np.random.choice(len(predicted_bboxes), size=n)

iou_to_draw = iou[indexes]
norm_to_draw = original_images[indexes]
display_digits_with_boxes(original_images[indexes], predicted_bboxes[indexes], normalized_bboxes[indexes], iou[indexes], "True and Predicted values", bboxes_normalized=True)

