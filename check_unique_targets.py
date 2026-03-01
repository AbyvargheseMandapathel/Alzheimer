import pandas as pd
try:
    df = pd.read_csv(r"c:\Users\ABY\Desktop\project\alzheimers\Data imputation KNN 2 neighbor\joined Labeled.csv")
    targets = df['Target'].unique().tolist()
    with open("unique_targets.txt", "w") as f:
        f.write(str(targets))
except Exception as e:
    with open("unique_targets.txt", "w") as f:
        f.write(str(e))
