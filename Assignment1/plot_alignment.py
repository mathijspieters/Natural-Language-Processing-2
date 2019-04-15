import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot(alignments, words1, words2, title="test"):

    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(alignments)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(words2)))
    ax.set_yticks(np.arange(len(words1)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(words2)
    ax.set_yticklabels(words1)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(words2)):
        for j in range(len(words1)):
            text = ax.text(i, j, "%.1f " % (10*alignments[j, i]),
                            ha="center", va="center", color="w")
            

    ax.set_title("Confusion matrix")
    plt.xlabel('French')
    plt.ylabel('English')
    fig.savefig('results/%s.png' % title, dpi=300)