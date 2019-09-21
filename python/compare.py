import pandas as pd

df1 = pd.read_csv("../train.csv").iloc[0:1, :5]
df2 = pd.read_csv("../train.csv").iloc[1:3, :]
print(df1)
print("----")
print(df2)

# print(pd.concat([df1,df2]).drop_duplicates(keep=False))

print("~~~~")
print(df1.shape, df2.shape)

df1_size = df1.shape[0]*df1.shape[1]
df2_size = df2.shape[0]*df2.shape[1]
if df1_size > df2_size:
    print('df1 bigger')
    # Find Rows in DF1 Which Are Not Available in DF2
    df = df1.merge(df2, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='left_only']
elif df1_size < df2_size:
    print('df2 bigger')
    # Find Rows in DF2 Which Are Not Available in DF1
    df = df2.merge(df1, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='left_only']
    # df = df1.merge(df2, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='right_only']
else:
    print('df1 and df2 same')
    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)
    df_gpby = df.groupby(list(df.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    df = df.reindex(idx)

print(df)
print('row diff:', df.shape[0], 'col diff:', df.shape[1])

# print(set(df1).symmetric_difference(df2))