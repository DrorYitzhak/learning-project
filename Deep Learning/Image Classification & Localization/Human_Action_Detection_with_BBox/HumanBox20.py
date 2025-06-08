import zipfile
from pathlib import Path
import io
from PIL import Image
import xml.etree.ElementTree as ET

# הנתיב לקובץ ה-ZIP
zip_path = Path(r"C:/Users/drory/Downloads/Stanford_40_full.zip")

# פתיחת הקובץ לבדיקה
with zipfile.ZipFile(zip_path, 'r') as zf:
    # הצגת קבצים בתיקייה Annotations
    annotation_files = [f for f in zf.namelist() if f.startswith("Annotations/") and f.endswith(".xml")]
    print(f"📄 נמצאו {len(annotation_files)} קבצי אנוטציה")

    # בודק אם קיימת תמונה לדוגמה ואנוטציה תואמת
    example_ann = annotation_files[0]
    example_img = example_ann.replace("Annotations", "JPEGImages").replace(".xml", ".jpg")

    # טוען את התמונה מה־ZIP
    with zf.open(example_img) as img_file:
        image = Image.open(img_file)
        image.show(title="תמונה לדוגמה")

    # טוען את קובץ ה-XML
    with zf.open(example_ann) as ann_file:
        tree = ET.parse(ann_file)
        root = tree.getroot()
        print(f"\n📦 תיבת סימון עבור: {example_img}")
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            print(f"- {name}: [{xmin}, {ymin}, {xmax}, {ymax}]")
