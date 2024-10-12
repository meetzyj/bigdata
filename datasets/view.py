import pandas as pd

df = pd.read_csv('./train.csv')
print('共{}个用户，{}本图书，{}条记录'.format(max(df['user_id'])+1, max(df['item_id'])+1, len(df)))
print(df.head())