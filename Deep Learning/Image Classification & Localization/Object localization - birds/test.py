# ✅ Object Localization + Classification (CUB-200-2011)
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score

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

class_names = {int(line.split()[0]): line.split()[1].split('.')[-1].replace('_', ' ') for line in class_lines}

images_dict = {}
for img_line, bbox_line, split_line, label_line in zip(image_lines, bbox_lines, split_lines, label_lines):
    img_id, img_path = img_line.strip().split()
    _, x, y, w, h = map(float, bbox_line.strip().split())
    _, is_train = split_line.strip().split()
    _, label = label_line.strip().split()

    images_dict[int(img_id)] = {
        "path": os.path.join(images_folder, img_path),
        "bbox": (x, y, x + w, y + h),
        "label": int(label) - 1,
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
def preprocess(image, label_bbox):
    label, bbox = label_bbox
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 127.5 - 1
    return image, (tf.one_hot(label, 200), bbox)

BATCH_SIZE = 32

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

    class_output = tf.keras.layers.Dense(200, activation='softmax', name='class_output')(x)
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
train_size = len([v for v in images_dict.values() if v["is_train"]])
val_size = len([v for v in images_dict.values() if not v["is_train"]])
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5
)

# ---------- גרפים ----------
def plot_metric(metric):
    plt.plot(history.history[metric], label=f"train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
    plt.title(metric)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_metric("class_output_accuracy")
plot_metric("bbox_output_loss")




# ---------- חישוב IoU ----------
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

# ---------- המרה לפיקסלים ----------
def denormalize(bbox, image_shape):
    h, w = image_shape[:2]
    ymin, xmin, ymax, xmax = bbox
    return [int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)]

# ---------- הצגת תחזיות ----------
def visualize_predictions(model, dataset, class_names, num_samples=10):
    y_true_cls, y_pred_cls = [], []
    y_true_bbox, y_pred_bbox = [], []
    iou_scores = []
    images_to_plot = []

    for images, (labels, bboxes) in dataset.take(1):
        preds_cls, preds_bbox = model.predict(images)

        for i in range(num_samples):
            image = ((images[i].numpy() + 1) * 127.5).astype(np.uint8)
            true_class = tf.argmax(labels[i]).numpy()
            pred_class = np.argmax(preds_cls[i])
            true_name = class_names[true_class + 1]
            pred_name = class_names[pred_class + 1]
            iou = calculate_iou(denormalize(bboxes[i], image.shape), denormalize(preds_bbox[i], image.shape))

            img_disp = image.copy()
            cv2.rectangle(img_disp, tuple(denormalize(bboxes[i], image.shape)[1::-1]),
                          tuple(denormalize(bboxes[i], image.shape)[3:1:-1]), (0, 255, 0), 2)
            cv2.rectangle(img_disp, tuple(denormalize(preds_bbox[i], image.shape)[1::-1]),
                          tuple(denormalize(preds_bbox[i], image.shape)[3:1:-1]), (255, 0, 0), 2)

            label = '🟢 Correct' if true_class == pred_class else '🔴 Wrong'
            title = f"{label}\n🐦 True: {true_name}\n🔮 Pred: {pred_name}\n📊 IoU: {iou:.2f}"
            images_to_plot.append((img_disp, title))

            y_true_cls.append(true_class)
            y_pred_cls.append(pred_class)
            y_true_bbox.append(bboxes[i].numpy())
            y_pred_bbox.append(preds_bbox[i])

    # ---------- הצגת תמונות ----------
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    for idx, (img, title) in enumerate(images_to_plot):
        ax = axes[idx // cols, idx % cols]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    for idx in range(len(images_to_plot), rows * cols):
        axes[idx // cols, idx % cols].axis('off')
    plt.tight_layout()
    plt.show()

    # ---------- מדדי סיווג ----------
    print("\n📋 Classification Report:")
    unique_labels = sorted(set(y_true_cls + y_pred_cls))
    target_names = [class_names[i + 1] for i in unique_labels]
    print(classification_report(y_true_cls, y_pred_cls, labels=unique_labels, target_names=target_names))

    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ---------- מדדי רגרסיה (מיקום) ----------
    print("\n📈 Bounding Box Evaluation:")
    avg_iou = np.mean(iou_scores)
    bbox_r2 = r2_score(np.array(y_true_bbox), np.array(y_pred_bbox))
    classification_acc = accuracy_score(y_true_cls, y_pred_cls)

    print(f"Average IoU: {avg_iou:.3f}")
    print(f"R² Score (Bounding Boxes): {bbox_r2:.3f}")
    print(f"Accuracy (Classification): {classification_acc:.3f}")

# ---------- הרצת שלב תחזית ----------
visualize_predictions(model, validation_dataset, class_names, num_samples=10)


