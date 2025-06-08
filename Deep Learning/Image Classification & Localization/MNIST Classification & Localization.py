import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

graphviz_path = "C:\\Program Files\\Graphviz-12.2.1-win64\\bin"
os.environ["PATH"] += os.pathsep + graphviz_path

print("Tensorflow version " + tf.__version__)

# @title Plot Utilities for Bounding Boxes
im_width = 75
im_height = 75
use_normalized_coordinates = True


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color=[],
                                       thickness=1,
                                       display_str_list=()):
    """Draws bounding boxes on image (numpy array).
    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: a list of strings for each bounding box.
    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)
    draw_bounding_boxes_on_image(rgbimg, boxes, color, thickness,
                                 display_str_list)
    return np.array(rgbimg)


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color=[],
                                 thickness=1,
                                 display_str_list=()):
    """Draws bounding boxes on image.
    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: a list of strings for each bounding box.

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
                                   boxes[i, 2], color[i], thickness, display_str_list[i])


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=1,
                               display_str=None,
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.
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
      display_str_list: string to display in box
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)


# @title Visualization Utilities
"""
This cell contains helper functions used for visualization
and downloads only. 

You can skip reading it, as there is very
little Keras or Tensorflow related code here.
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


# pull a batch from the datasets.
def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    # get one batch from each: 10000 validation digits, N training digits
    batch_train_ds = training_dataset.unbatch().batch(N)

    for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:
        validation_digits = validation_digits.numpy()
        validation_labels = validation_labels.numpy()
        validation_bboxes = validation_bboxes.numpy()
        break
    for training_digits, (training_labels, training_bboxes) in batch_train_ds:
        training_digits = training_digits.numpy()
        training_labels = training_labels.numpy()
        training_bboxes = training_bboxes.numpy()
        break

    # these were one-hot encoded in the dataset
    validation_labels = np.argmax(validation_labels, axis=1)
    training_labels = np.argmax(training_labels, axis=1)

    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)


# utility to display a row of digits with their predictions
def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 10

    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]

    if (len(pred_bboxes) > 0):
        n_pred_bboxes = pred_bboxes[indexes, :]

    if (len(bboxes) > 0):
        n_bboxes = bboxes[indexes, :]

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(n, 75, 75)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if (len(pred_bboxes) > i):
            bboxes_to_plot.append(n_pred_bboxes[i])

        if (len(bboxes) > i):
            bboxes_to_plot.append(n_bboxes[i])

        img_to_draw = draw_bounding_boxes_on_image_array(image=n_digits[i], boxes=np.asarray(bboxes_to_plot),
                                                         color=['red', 'green'], display_str_list=["true", "pred"])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if (n_iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (n_iou[i][0]), color=color, transform=ax.transAxes)


# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)



BATCH_SIZE = 64



"""
Loading and Preprocessing the Dataset
Define some helper functions that will pre-process your data:

read_image_tfds: randomly overlays the "digit" image on top of a larger canvas.
get_training_dataset: loads data and splits it to get the training set.
get_validation_dataset: loads and splits the data to get the validation set.
"""

'''
Transforms each image in dataset by pasting it on a 75x75 canvas at random locations.
'''


def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (28, 28, 1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)

    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])


'''
Loads and maps the training split of the dataset using the map function. 
'''


def get_training_dataset():
    dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(
        -1)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    return dataset


'''
Loads and maps the validation split of the dataset using the map function. 
'''


def get_validation_dataset():
    dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    # dataset = dataset.cache() # this small dataset can be entirely cached in RAM
    dataset = dataset.batch(10000)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    return dataset


# instantiate the datasets
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()



"""Visualize Data"""
(training_digits, training_labels, training_bboxes,
 validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset, 10)

display_digits_with_boxes(training_digits, training_labels, training_labels, np.array([]), training_bboxes, np.array([]), "training digits and their labels")
display_digits_with_boxes(validation_digits, validation_labels, validation_labels, np.array([]), validation_bboxes, np.array([]), "validation digits and their labels")





"""
Define the Model
Here, you'll define your custom CNN.

feature_extractor: these convolutional layers extract the features of the image.
classifier: This define the output layer that predicts among 10 categories (digits 0 through 9)
bounding_box_regression: This defines the output layer that predicts 4 numeric values, which define the coordinates of the bounding box (xmin, ymin, xmax, ymax)
final_model: This combines the layers for feature extraction, classification and bounding box prediction.
Notice that this is another example of a branching model, because the model splits to produce two kinds of output (a category and set of numbers).
Since you've learned to use the Functional API earlier in the specialization (course 1), you have the flexibility to define this kind of branching model!
define_and_compile_model: choose the optimizer and metrics, then compile the model.
"""

'''
Feature extractor is the CNN that is made up of convolution and pooling layers.
'''


def feature_extractor(inputs):
    # x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(inputs)
    x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3)(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    return x


'''
dense_layers adds a flatten and dense layer.
This will follow the feature extraction layers
'''


def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x


'''
Classifier defines the classification output.
This has a set of fully connected layers and a softmax layer.
'''


def classifier(inputs):
    classification_output = tf.keras.layers.Dense(10, activation='softmax', name='classification')(inputs)
    return classification_output


'''
This function defines the regression output for bounding box prediction. 
Note that we have four outputs corresponding to (xmin, ymin, xmax, ymax)
'''


def bounding_box_regression(inputs):
    # bounding_box_regression_output = tf.keras.layers.Dense(units='4', name='bounding_box')(inputs)
    bounding_box_regression_output = tf.keras.layers.Dense(units=4, name='bounding_box')(inputs)

    return bounding_box_regression_output


def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)

    '''
    The model branches here.  
    The dense layer's output gets fed into two branches:
    classification_output and bounding_box_output
    '''
    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)

    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])

    return model


def define_and_compile_model(inputs):
    model = final_model(inputs)

    model.compile(optimizer='adam',
                  loss={'classification': 'categorical_crossentropy',
                        'bounding_box': 'mse'
                        },
                  metrics={'classification': 'accuracy',
                           'bounding_box': 'mse'
                           })
    return model


inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
model = define_and_compile_model(inputs)

# print model layers
# tf.keras.utils.plot_model(model, show_shapes=True)
tf.keras.utils.plot_model(model, show_shapes=True, to_file="model_structure.png")


"""
Train and validate the model
Train the model.

You can choose the number of epochs depending on the level of performance
"""
EPOCHS = 10
# EPOCHS = 6
steps_per_epoch = 60000//BATCH_SIZE  # 60,000 items in this dataset
validation_steps = 10000//BATCH_SIZE  # 10,000 items in this dataset

history = model.fit(training_dataset,
                    steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps, epochs=EPOCHS)

loss, classification_loss, bounding_box_loss, classification_accuracy, bounding_box_mse = model.evaluate(validation_dataset, steps=1)
print("Validation accuracy: ", classification_accuracy)


plot_metrics("classification_loss", "Classification Loss", 1.5)
plot_metrics("bounding_box_loss", "Bounding Box Loss", 1.5)


"""
Intersection over union
Calculate the I-O-U metric to evaluate the model's performance.
"""
def intersection_over_union(pred_box, true_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)

    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0) * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou




"""
Visualize predictions
The following code will make predictions and visualize both the classification and the predicted bounding boxes.

The true bounding box labels will be in green, and the model's predicted bounding boxes are in red.
The predicted number is shown below the image.
"""

# recognize validation digits
predictions = model.predict(validation_digits, batch_size=64)
predicted_labels = np.argmax(predictions[0], axis=1)

predicted_bboxes = predictions[1]

iou = intersection_over_union(predicted_bboxes, validation_bboxes)

iou_threshold = 0.6

print("Number of predictions where iou > threshold(%s): %s" % (iou_threshold, (iou >= iou_threshold).sum()))
print("Number of predictions where iou < threshold(%s): %s" % (iou_threshold, (iou < iou_threshold).sum()))


display_digits_with_boxes(validation_digits, predicted_labels, validation_labels, predicted_bboxes, validation_bboxes, iou, "True and Predicted values")
pass

# ================================================= Grad-CAM ==========================================================

"""
🧠 What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is a method for visualizing which parts of an image
were important for a convolutional neural network (CNN) when making a classification decision.

How it works:
1. It looks at the last convolutional layer of the CNN.
2. It calculates the gradient of the predicted class score with respect to each feature map.
3. It uses those gradients to weigh the importance of each feature map.
4. Then it combines them into a heatmap that shows the "attention" of the model.

This is useful for understanding and debugging CNN models.
"""

import cv2
import matplotlib.cm as cm

# Grad-CAM function (unchanged)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[0]]
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

# Number of images to visualize
num_images = 5

plt.figure(figsize=(15, num_images * 3))

for i in range(num_images):
    img = validation_digits[i]  # Shape: (75, 75)
    img_input = np.expand_dims(np.expand_dims(img, axis=-1), axis=0).astype(np.float32)  # (1, 75, 75, 1)

    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name="conv2d_2")

    # Convert to RGB for display
    img_orig = np.uint8(img * 255.0)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_color = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_color = np.uint8(255 * heatmap_color)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    # Plotting
    plt.subplot(num_images, 3, 3 * i + 1)
    plt.title(f"Original #{i}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(num_images, 3, 3 * i + 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.subplot(num_images, 3, 3 * i + 3)
    plt.title("Superimposed")
    plt.imshow(superimposed_img)
    plt.axis('off')

plt.tight_layout()
plt.show()