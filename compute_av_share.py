import pandas as pd

df = pd.read_csv("combine_av.csv")

# Total AV per draft class
class_totals = df.groupby('draft_year')['car_av'].sum().reset_index()
class_totals.columns = ['draft_year', 'total_class_av']

# Merge with original dataframe
df = df.merge(class_totals, on='draft_year', how='left')

# AV share calculation - car AV / total class AV
df['av_share'] = df['car_av'] / df['total_class_av']

df.to_csv("combine_av_share.csv", index=False) # Save to new csv

print(df.head())

