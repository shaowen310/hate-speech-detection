# %%
import seaborn as sbn
import matplotlib.pyplot as plt

# %%
from dataprovider.twitter_hatred_speech import TwitterHatredSpeech

twitter_dataset = TwitterHatredSpeech(data_dir="data/twitter_hatred_speech")

# %%
df_train = twitter_dataset.train_split()

print(f"Columns: {df_train.columns}")

# %%
# Check distribution of labels
sbn.set(style="whitegrid")
plt.figure(figsize=(6, 2))

plt.subplot(1, 2, 1)
# using the bully_pallete function to create a custom pallete
sbn.countplot(x="label", data=df_train)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Labels")

## Found twitter_hatred_speech training data to be extremely imbalanced


# %%
