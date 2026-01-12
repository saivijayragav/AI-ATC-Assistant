import pandas as pd
df_backup = pd.read_csv("states_2022-01-03-00.csv")
df_backup = df_backup.sample(2000, random_state=42)
df_backup.to_csv("backend/backup_snapshot.csv", index=False)
