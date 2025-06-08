
# ✅ Object Localization + Classification (Top 20 Classes from CUB-200-2011)
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score
from collections import Counter
import math

# ---------- נתיבים ----------
dataset_root = r"C:\Users\drory\Downloads\200_Bird_Dataset\CUB_200_2011\CUB_200_2011"
images_folder = os.path.join(dataset_root, "images")
images_txt = os.path.join(dataset_root, "images.txt")
bboxes_txt = os.path.join(dataset_root, "bounding_boxes.txt")
split_txt = os.path.join(dataset_root, "train_test_split.txt")
labels_txt = os.path.join(dataset_root, "image_class_labels.txt")
classes_txt = os.path.join(dataset_root, "classes.txt")

# ---------- טעינת קבצים ----------
with open(images_txt, 'r') as f:
    image_lines = f.read().strip().split('\n')
with open(bboxes_txt, 'r') as f:
    bbox_lines = f.read().strip().split('\n')
with open(split_txt, 'r') as f:
    split_lines = f.read().strip().split('\n')
with open(labels_txt, 'r') as f:
    label_lines = f.read().strip().split('\n')
with open(classes_txt, 'r') as f:
    class_lines = f.read().strip().split('\n')

class_id_to_name = {int(line.split()[0]): line.split()[1].split('.')[-1].replace('_', ' ') for line in class_lines}
label_counts = Counter([int(line.split()[1]) for line in label_lines])
top_20_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]
top_20_ids = set([item[0] for item in top_20_classes])
id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(top_20_ids))}
class_names = {id_map[old_id]: class_id_to_name[old_id] for old_id in id_map}

images_dict = {}
for img_line, bbox_line, split_line, label_line in zip(image_lines, bbox_lines, split_lines, label_lines):
    parts = img_line.strip().split()
    if len(parts) < 2:
        continue
    img_id, img_path = parts
    x, y, w, h = map(float, bbox_line.strip().split()[1:])
    is_train = split_line.strip().split()[-1]
    label = int(label_line.strip().split()[-1])

    if label not in top_20_ids:
        continue

    new_label = id_map[label]
    images_dict[int(img_id)] = {
        "path": os.path.join(images_folder, img_path),
        "bbox": (x, y, x + w, y + h),
        "label": new_label,
        "is_train": is_train == '1'
    }

# ---------- Dataset Generator ----------
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

# ---------- Preprocessing ----------
NUM_CLASSES = 20
BATCH_SIZE = 32

def preprocess(image, label_bbox):
    label, bbox = label_bbox
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 127.5 - 1
    return image, (tf.one_hot(label, NUM_CLASSES), bbox)

def get_dataset(dataset, training=True):
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(512).repeat()
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

raw_training_dataset = load_bird_dataset(images_dict, is_train=True)
raw_validation_dataset = load_bird_dataset(images_dict, is_train=False)
training_dataset = get_dataset(raw_training_dataset, training=True)
validation_dataset = get_dataset(raw_validation_dataset, training=False)

# ---------- מודל ----------
def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    class_output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)
    bbox_output = tf.keras.layers.Dense(4, name='bbox_output')(x)

    model = tf.keras.Model(inputs, outputs=[class_output, bbox_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mse'},
        metrics={'class_output': 'accuracy', 'bbox_output': 'mse'}
    )
    return model

model = build_model()
model.summary()

# ---------- אימון ----------

train_size = sum(1 for v in images_dict.values() if v["is_train"])
val_size = sum(1 for v in images_dict.values() if not v["is_train"])

steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
validation_steps = math.ceil(val_size / BATCH_SIZE)


print(f"🧮 Train size: {train_size}, Steps per epoch: {steps_per_epoch}")
print(f"🧮 Val size: {val_size}, Validation steps: {validation_steps}")

model.fit(
    training_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5
)

# ---------- הערכה ----------
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
        print(f"Batch size: {images.shape[0]}, Labels: {labels.shape}, BBoxes: {bboxes.shape}, Preds_bbox: {preds_bbox.shape}")

        for i in range(images.shape[0]):
            image = ((images[i].numpy() + 1) * 127.5).astype(np.uint8)
            true_class = tf.argmax(labels[i]).numpy()
            pred_class = np.argmax(preds_cls[i])
            iou = calculate_iou(denormalize(bboxes[i], image.shape), denormalize(preds_bbox[i], image.shape))
            iou_scores.append(iou)
            y_true_cls.append(true_class)
            y_pred_cls.append(pred_class)
            y_true_bbox.append(bboxes[i].numpy())
            y_pred_bbox.append(preds_bbox[i])
            image_predictions.append({
                'image': image,
                'true_class': true_class,
                'pred_class': pred_class,
                'iou': iou,
                'true_bbox': bboxes[i].numpy(),
                'pred_bbox': preds_bbox[i]
            })

    print("\n📋 Classification Report:")
    unique_labels = sorted(set(y_true_cls + y_pred_cls))
    target_names = [class_names[i] for i in unique_labels]
    print(classification_report(y_true_cls, y_pred_cls, labels=unique_labels, target_names=target_names))

    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\n📈 Bounding Box Evaluation:")
    print(f"Average IoU: {np.mean(iou_scores):.3f}")
    print(f"R² Score (Bounding Boxes): {r2_score(np.array(y_true_bbox), np.array(y_pred_bbox)):.3f}")
    print(f"Accuracy (Classification): {accuracy_score(y_true_cls, y_pred_cls):.3f}")

    print(f"\n🖼️ Displaying {display_samples} random sample predictions:")
    cols = 5
    rows = (display_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    selected = random.sample(image_predictions, display_samples)
    for idx, sample in enumerate(selected):
        img = sample['image'].copy()
        cv2.rectangle(img, tuple(denormalize(sample['true_bbox'], img.shape)[1::-1]),
                      tuple(denormalize(sample['true_bbox'], img.shape)[3:1:-1]), (0, 255, 0), 2)
        cv2.rectangle(img, tuple(denormalize(sample['pred_bbox'], img.shape)[1::-1]),
                      tuple(denormalize(sample['pred_bbox'], img.shape)[3:1:-1]), (255, 0, 0), 2)
        ax = axes[idx // cols, idx % cols]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = 'green' if sample['true_class'] == sample['pred_class'] else 'red'
        title = f"IoU: {sample['iou']:.2f}\nTrue: {class_names[sample['true_class']]}\nPred: {class_names[sample['pred_class']]}"
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    for idx in range(len(selected), rows * cols):
        axes[idx // cols, idx % cols].axis('off')
    plt.tight_layout()
    plt.show()

# ---------- הרצה ----------
evaluate_full_validation(model, validation_dataset, class_names)
