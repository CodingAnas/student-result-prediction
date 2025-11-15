import os

#Just gets data and save it in cache
import kagglehub
path = kagglehub.dataset_download("himanshunakrani/student-study-hours")

import pandas as pd
df = pd.read_csv(path+"/score_updated.csv")
df = df.rename(columns={"Hours":"study_hours"})
df = df.rename(columns={"Scores":"scores"})

#Generating synthetic sleep hours
import numpy as np
df['sleep_hours'] = (
    8                    
    - (df['study_hours'] * 0.2)   
    + (df['scores'] * 0.01)  
    + np.random.normal(0, 0.8, len(df))
)
df['sleep_hours'] = df['sleep_hours'].clip(4, 10).round(1)

#saving it in data directory
os.makedirs("data", exist_ok=True)
df.to_csv("./data/student_scores.csv", index=False)