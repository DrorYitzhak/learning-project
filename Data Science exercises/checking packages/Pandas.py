import pandas as pd
df = pd.read_csv('C:\\Users\\drory\\Desktop\\Data for practice\\Lab data for testing.csv')
print(df.head(3))



df['Test Name']  # Everything under the title (the entire column)
df.iloc[1:5,0:3] # df.iloc[columns,rows]
# df.iloc[1,2] = "A value that we want to replace or put in this cell"
df.to_csv("The_name_of_the_file_we_want_to_save_after_modification.csv")
