import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# נתוני דוגמה (במקום זה יש לטעון את הנתונים האמיתיים שלך)
np.random.seed(0)
torch.manual_seed(0)

# יצירת נתוני אימון לדוגמה
num_samples = 200
num_features = 8  # 4 מתחים + 4 זרמים (סך הכל 8 מאפיינים)
num_rf_metrics = 3  # Gain_mm_dB, eirp, imrr

# יצירת נתונים אקראיים
x_train = torch.tensor(np.random.rand(num_samples, num_features), dtype=torch.float32)  # מתחים וזרמים
y_train = torch.tensor(np.random.rand(num_samples, num_rf_metrics), dtype=torch.float32)  # תוצאות RF

# יצירת DataLoader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# הגדרת המודל
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, num_rf_metrics)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# יצירת אובייקט של המודל
model = RegressionModel()

# הגדרת פונקציית אובדן ומבצע
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# אימון המודל
num_epochs = 100
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # אפס את הגרדיאנטים
        optimizer.zero_grad()

        # חיזוי הערכים
        outputs = model(x_batch)

        # חישוב האובדן
        loss = criterion(outputs, y_batch)

        # חישוב הגרדיאנטים וביצוע צעד אופטימיזציה
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# חישוב חשיבות המאפיינים עבור כל מדידה
model.eval()
x_train.requires_grad = True  # מאפשר חישוב גרדיאנטים
outputs = model(x_train)

# חישוב גרדיאנטים עבור כל מדידה
feature_importance = np.zeros((num_rf_metrics, num_features))
for i in range(num_rf_metrics):
    sample_output = outputs[:, i].mean()  # ממוצע של כל הפלטים של מדידה ספציפית
    sample_output.backward(retain_graph=True)  # שומרים את הגרף

    # קבלת הגרדיאנטים של המאפיינים
    feature_importance[i, :] = x_train.grad.abs().mean(dim=0).numpy()

    # איפוס הגרדיאנטים של x_train
    x_train.grad.zero_()

# הצגת ההשפעה של כל מאפיין עבור כל מדידה של RF
feature_names = [
    'VDD 1.8V', 'VDD 1.25V', 'VDD 0.95V', 'VDD 0.8V',
    'CMU1 LDO Clk', 'CMU1 LDO DAC Ana', 'CMU1 LDO DAC Dig', 'CMU2 LDO BG'
]

for i, metric in enumerate(['Gain_mm_dB', 'eirp', 'imrr']):
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), feature_importance[i, :], tick_label=feature_names)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance for {metric}')
    plt.xticks(rotation=45, ha='right')  # לסובב את התוויות כדי שיהיה יותר קריא
    plt.tight_layout()
    plt.show()

    print(f"Feature importances for {metric}:")
    for name, importance in zip(feature_names, feature_importance[i, :]):
        print(f"{name}: {importance:.4f}")
