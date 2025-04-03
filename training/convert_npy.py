import numpy as np
import json

labels = np.load("exercise_labels.npy")
with open("public/exercise_labels.json", "w") as f:
    json.dump(labels.tolist(), f)
print("✅ Converted to exercise_labels.json")
