import pandas as pd

# Gesamt-CSV laden
df = pd.read_csv("data/VQA-Rad/vqarad_annotations.csv")

# Für jeden Split eine eigene Datei erzeugen
for split in ["train", "validation", "test"]:
    df_split = df[df["split"] == split]
    df_split = df_split[["split", "image_path", "question", "answer"]]
    out_file = f"data/VQA-Rad/vqarad_{split}.csv"
    df_split.to_csv(out_file, index=False)
    print(f"{out_file} geschrieben, {len(df_split)} Einträge.")
