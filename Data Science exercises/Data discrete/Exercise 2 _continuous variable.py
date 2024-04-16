import pandas as pd
import matplotlib as mplf
from matplotlib import pyplot as plt
df = pd.read_csv('C:\\Users\\drory\\Desktop\\Data for practice\\Bike_sharing_data.csv')
# print(df.head(3))
# print(df.info())
print(pd.cut(df["temp"],bins=5).value_counts())  # חלוקה של המספקים הרציפים לפי bins, והוספת value_counts כדי לדעת כמה ערכים יש בכל מחלקה
print(df.cnt.describe())  # cnt מדדי המרכז של
print(plt.hist(df["temp"],bins=5))  # הצגת איסטוגרמה