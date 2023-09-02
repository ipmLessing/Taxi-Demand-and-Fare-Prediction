import pandas as pd
data = pd.read_csv("file.csv", sep=",")
print(data.head(2))
with open('file.json', 'w') as f:
    f.write(data.to_json(orient='records', lines=True))

# check
data = pd.read_json("file.json", lines=True)
print(data.head(2))