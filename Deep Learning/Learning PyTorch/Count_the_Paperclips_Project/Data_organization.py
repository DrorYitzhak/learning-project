import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class PaperclipsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'clips-{self.data_frame.iloc[idx, 0]}.png')
        image = Image.open(img_name)

        # Convert the image to grayscale
        image = image.convert('L')

        clip_count = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, clip_count

if __name__ == '__main__':
    # הגדר את נתיבי הקבצים
    data_dir = r'C:\Users\drory\learning-project\Deep Learning\Learning PyTorch\Count_the_Paperclips_Project\data'
    clips_data_dir = os.path.join(data_dir, 'clips_data_2020', 'clips')
    train_csv_path = os.path.join(data_dir, 'train.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv')

    # הגדר טרנספורמציות לתמונות
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    # צור את ה-Datasets
    train_dataset = PaperclipsDataset(csv_file=train_csv_path, root_dir=clips_data_dir, transform=transform)
    test_dataset = PaperclipsDataset(csv_file=test_csv_path, root_dir=clips_data_dir, transform=transform)

    # צור את ה-DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

    # דוגמה לאיטרציה על ה-DataLoader
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        # הוסף כאן את הקוד להכנסת הנתונים למודל שלך
        break

    # ביטול מצב אינטראקטיבי של matplotlib
    plt.ioff()

    # הצגת תמונה מתוך קבוצת האימון
    def imshow(img, title):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.show()

    # הצגת התמונה הראשונה מהאימון
    dataiter = iter(train_loader)
    images, labels = next(dataiter)  # שימוש בפונקציה המובנית next()
    imshow(images[0], f'Train Image - Label: {labels[0]}')

    # הצגת התמונה הראשונה מהמבחן
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  # שימוש בפונקציה המובנית next()
    imshow(images[0], f'Test Image - Label: {labels[0]}')
