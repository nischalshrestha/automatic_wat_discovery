import pandas as pd


# index = [0,1]
index = ['a','b']
df = pd.DataFrame([[1,2],[3,4]], index=index, columns=['a','b'])
start = 0
end = 2
# print(df.loc[f'{int(start)}:{int(end)}', 'a'])

col = "a"

print(df[0::2])

# print(df.columns.values)
# for i in df.columns:
#     print(i)
# print(df.loc[f'{col}'])

# print(df.index.values)

# index is rownames basically, where default is int slice
# print(df.loc[0:2,:])
# print(df[0:2])

# print(df)
# print(df.loc['a', 'a':'b'])
