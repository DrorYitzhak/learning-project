import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv('C:\\Users\\drory\\Desktop\\Data for practice\\Bike_sharing_data.csv')
print(df.head(3))
print(df.info())
print(df['mnth'].value_counts()) # מצג את השכיחות
print(df['mnth'].value_counts(normalize=True)) # מצג את השכיחות היחסית לכלל המדגם
