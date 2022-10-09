import numpy as np
import matplotlib.pyplot as plt
import json

with open("./trained/char/scores.json") as file:
  char_scores = json.load(file)

with open("./trained/word/scores.json") as file:
  word_scores = json.load(file)

for scores, title, file_name in zip([char_scores, word_scores], 
                                    ["Character restoration training scores", "Word restoration training scores"],
                                    ["./plots/char_plot.pdf", "./plots/word_plot.pdf"]):
  target_scores = scores["target"]
  crpt_scores = scores["corrupt"]
  
  x_ticks = [len(s) for s in target_scores]
  x_ticks = [0] + [sum(x_ticks[:i + 1]) - 1 for i in range(len(x_ticks))]
  x_tick_labs = np.array([str(xtl) if i % 5 == 0 else "" for i, xtl in enumerate(range(len(x_ticks)))])
  
  target_scores = np.concatenate(target_scores)
  crpt_scores = np.concatenate(crpt_scores)
  
  fig, ax = plt.subplots(1, 1, figsize=(7,3.5), dpi=120)
  plt.plot(target_scores, label="distance to target images", linewidth=0.9)
  plt.plot(crpt_scores, label="distance to corrupt images", linewidth=0.9)
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_tick_labs)
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Mean absolute pixel distance")
  ax.set_title(title)
  plt.legend()
  plt.tight_layout()
  fig.savefig(file_name, format="pdf")
  plt.show()