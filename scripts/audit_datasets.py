import os
import glob
import pandas as pd

data_dir = "data"
ftv = os.path.join(data_dir, "Fundus_Train_Val_Data", "Fundus_Scanes_Sorted")
tp = len(glob.glob(os.path.join(ftv, "Train", "Glaucoma_Positive", "*.*")))
tn = len(glob.glob(os.path.join(ftv, "Train", "Glaucoma_Negative", "*.*")))
vp = len(glob.glob(os.path.join(ftv, "Validation", "Glaucoma_Positive", "*.*")))
vn = len(glob.glob(os.path.join(ftv, "Validation", "Glaucoma_Negative", "*.*")))
print(f"Fundus_Train_Val_Data: Train(Glaucoma={tp}, Normal={tn}) | Val(Glaucoma={vp}, Normal={vn}) | Total={tp+tn+vp+vn}")

acrima_dir = os.path.join(data_dir, "ACRIMA", "Images")
af = glob.glob(os.path.join(acrima_dir, "*.*"))
ap = len([f for f in af if "_g_" in os.path.basename(f).lower()])
an = len([f for f in af if "_g_" not in os.path.basename(f).lower()])
print(f"ACRIMA: Glaucoma={ap}, Normal={an} | Total={len(af)}")

csv_path = os.path.join(data_dir, "glaucoma.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    g_count = int((df["Glaucoma"] == 1).sum())
    n_count = int((df["Glaucoma"] == 0).sum())
    print(f"glaucoma.csv (ORIGA): Glaucoma={g_count}, Normal={n_count} | Total={len(df)}")

