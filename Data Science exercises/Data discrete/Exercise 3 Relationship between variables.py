import pandas as pd
import matplotlib as mplf
from matplotlib import pyplot as plt

df = pd.read_csv('C:\\Users\\drory\\Desktop\\Data for practice\\Bike_sharing_data.csv')
print(plt.scatter(df.registered, df.cnt))  # גרף המתאר את הקשר בין שתי משתנים (אם יוצא גרף לינארי נגיד שינשו קשר לינארי בין המשתנים )
print(df.cnt[df.cnt > 5000].count())  # כמות הערכים הקטנים מ5000 והצגת הכמות שלהם
print(df.cnt[df.cnt > 5000].count() / df.cnt.count())  #  ההסתברות למחסור באופנעים