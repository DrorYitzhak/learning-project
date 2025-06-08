from pathlib import Path


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score
import random

# ---------- נתיבים ----------
dataset_root = r"C:\\Users\\drory\\Downloads\\200_Bird_Dataset\\CUB_200_2011\\CUB_200_2011"
classes_txt = os.path.join(dataset_root, "classes.txt")
labels_txt = os.path.join(dataset_root, "image_class_labels.txt")
split_txt = os.path.join(dataset_root, "train_test_split.txt")
images_txt = os.path.join(dataset_root, "images.txt")
bboxes_txt = os.path.join(dataset_root, "bounding_boxes.txt")
images_folder = os.path.join(dataset_root, "images")

# ---------- טעינת קבצים ----------
with open(classes_txt, 'r') as f:
    class_lines = f.read().strip().split('\\n')
with open(labels_txt, 'r') as f:
    label_lines = f.read().strip().split('\\n')
with open(split_txt, 'r') as f:
    split_lines = f.read().strip().split('\\n')
with open(images_txt, 'r') as f:
    image_lines = f.read().strip().split('\\n')
with open(bboxes_txt, 'r') as f:
    bbox_lines = f.read().strip().split('\\n')

class_names = {int(line.split()[0]): line.split()[1].split('.')[-1].replace('_', ' ') for line in class_lines}

images_dict = {}
for img_line, bbox_line, split_line, label_line in zip(image_lines, bbox_lines, split_lines, label_lines):
    parts = img_line.strip().split()
    img_id, img_path = parts[0], " ".join(parts[1:])

    parts = bbox_line.strip().split()
    img_id = int(parts[0])
    x, y, w, h = map(float, parts[1:5])

    parts = split_line.strip().split()
    img_id = int(parts[0])
    is_train = parts[1]

    parts = label_line.strip().split()
    img_id = int(parts[0])
    label = parts[1]

    images_dict[int(img_id)] = {
        "path": os.path.join(images_folder, img_path),
        "bbox": (x, y, x + w, y + h),
        "label": int(label) - 1,
        "is_train": is_train == '1'
    }

def load_bird_dataset(images_dict, is_train=True):
    def _gen():
        for img_id in images_dict:
            if images_dict[img_id]["is_train"] != is_train:
                continue
            image_path = images_dict[img_id]["path"]
            label = images_dict[img_id]["label"]
            xmin, ymin, xmax, ymax = images_dict[img_id]["bbox"]

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            shape = tf.shape(image)
            height = tf.cast(shape[0], tf.float32)
            width = tf.cast(shape[1], tf.float32)

            bbox = [ymin / height, xmin / width, ymax / height, xmax / width]
            yield image, (label, tf.convert_to_tensor(bbox, dtype=tf.float32))

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            (tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(4,), dtype=tf.float32))
        )
    )

def preprocess(image, label_bbox):
    label, bbox = label_bbox
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 127.5 - 1
    return image, (tf.one_hot(label, 20), bbox)

def get_dataset(dataset, training=True):
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

validation_dataset = get_dataset(load_bird_dataset(images_dict, is_train=False), training=False)

def calculate_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    xi1 = max(xmin1, xmin2)
    yi1 = max(ymin1, ymin2)
    xi2 = min(xmax1, xmax2)
    yi2 = min(ymax1, ymax2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def denormalize(bbox, image_shape):
    h, w = image_shape[:2]
    ymin, xmin, ymax, xmax = bbox
    return [int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)]

def evaluate_full_validation(model, dataset, class_names, display_samples=10):
    y_true_cls, y_pred_cls = [], []
    y_true_bbox, y_pred_bbox = [], []
    iou_scores = []
    image_predictions = []

    for images, (labels, bboxes) in dataset:
        preds_cls, preds_bbox = model.predict(images)

        for i in range(images.shape[0]):
            image = ((images[i].numpy() + 1) * 127.5).astype(np.uint8)
            true_class = tf.argmax(labels[i]).numpy()
            pred_class = np.argmax(preds_cls[i])
            true_name = class_names[true_class + 1]
            pred_name = class_names[pred_class + 1]
            iou = calculate_iou(denormalize(bboxes[i], image.shape), denormalize(preds_bbox[i], image.shape))

            y_true_cls.append(true_class)
            y_pred_cls.append(pred_class)
            y_true_bbox.append(bboxes[i].numpy())
            y_pred_bbox.append(preds_bbox[i])
            iou_scores.append(iou)

            image_predictions.append({
                'image': image,
                'true_class': true_class,
                'pred_class': pred_class,
                'iou': iou,
                'true_bbox': bboxes[i].numpy(),
                'pred_bbox': preds_bbox[i]
            })

    print("\\n📋 Classification Report:")
    unique_labels = sorted(set(y_true_cls + y_pred_cls))
    target_names = [class_names[i + 1] for i in unique_labels]
    print(classification_report(y_true_cls, y_pred_cls, labels=unique_labels, target_names=target_names))

    print("\\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\\n📈 Bounding Box Evaluation:")
    avg_iou = np.mean(iou_scores)
    bbox_r2 = r2_score(np.array(y_true_bbox), np.array(y_pred_bbox))
    classification_acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"R² Score (Bounding Boxes): {bbox_r2:.3f}")
    print(f"Accuracy (Classification): {classification_acc:.3f}")

    print(f"\\n🖼️ Displaying {display_samples} random sample predictions:")
    selected = random.sample(image_predictions, display_samples)
    cols = 5
    rows = (display_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

    for idx, sample in enumerate(selected):
        img = sample['image'].copy()
        iou = sample['iou']
        true_cls = sample['true_class']
        pred_cls = sample['pred_class']
        true_bbox = sample['true_bbox']
        pred_bbox = sample['pred_bbox']

        cv2.rectangle(img, tuple(denormalize(true_bbox, img.shape)[1::-1]),
                      tuple(denormalize(true_bbox, img.shape)[3:1:-1]), (0, 255, 0), 2)
        cv2.rectangle(img, tuple(denormalize(pred_bbox, img.shape)[1::-1]),
                      tuple(denormalize(pred_bbox, img.shape)[3:1:-1]), (255, 0, 0), 2)

        ax = axes[idx // cols, idx % cols]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        title_color = 'green' if true_cls == pred_cls else 'red'
        true_name = class_names[true_cls + 1]
        pred_name = class_names[pred_cls + 1]
        title = f"IoU: {iou:.2f}\\nTrue: {true_name}\\nPred: {pred_name}"
        ax.set_title(title, fontsize=9, color=title_color)
        ax.axis('off')

    for idx in range(len(selected), rows * cols):
        axes[idx // cols, idx % cols].axis('off')

    plt.tight_layout()
    plt.show()

# ---------- טעינת המודל ----------
model = tf.keras.models.load_model("trained_model.h5")
evaluate_full_validation(model, validation_dataset, class_names)


output_path = Path("/mnt/data/post_eval.py")
output_path.write_text(code)

output_path
