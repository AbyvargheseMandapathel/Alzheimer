import pandas as pd

def inspect():
    try:
        df = pd.read_csv(r"c:\Users\ABY\Desktop\project\alzheimers\Data imputation KNN 2 neighbor\joined Labeled.csv")
        print("Columns:", df.columns.tolist())
        if 'CDR' in df.columns or 'CDRGLOB' in df.columns:
             print("Found CDR column!")
             print(df['CDRGLOB'].value_counts() if 'CDRGLOB' in df.columns else df['CDR'].value_counts())
        
        print("Target value counts:")
        print(df['Target'].value_counts())

    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect()
