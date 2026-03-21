import os
import re
import numpy as np
from collections import defaultdict

class DataLoader:
  def __init__(self, text, dictionary=None):

    self.text = np.array(re.sub('[^a-zA-Z]',' ', text).lower().split())

    word_freq = defaultdict(int)
    for word in self.text:
        word_freq[word] +=1
    
    if dictionary is None:
      dictionary = {}
      index = 0
      for word, count in word_freq.items():
        if count >= 3:
           dictionary[word] = index
           index += 1


    self.dictionary = dictionary
    self.dictionary_size = len(dictionary)
    self.reverse_dictionary = {v:k for k,v in self.dictionary.items()}

    t = 1e-5
    subsampled_tokens = []

    for w in self.text:
      if w not in dictionary:
        continue

      f = word_freq[w]
      p_keep = min(1.0, (t / f)**0.5 + t / f)

      if np.random.randn() < p_keep:
        subsampled_tokens.append(dictionary[w])

    self.tokenized_text = np.array(subsampled_tokens)

    #self.tokenized_text = np.array([dictionary[w] for w in self.text if w in dictionary])
    self.num_samples = None

  def load(self, window_size, batch_size):

    tokenized_text = self.tokenized_text
    ray = (window_size-1)//2

    contexts = []
    targets = []

    for i in range(ray, len(tokenized_text) - ray):
      context = []
      for j in range(i-ray,i+ray+1):
        if j != i:
          context.append(tokenized_text[j])
      contexts.append(context)
      targets.append(tokenized_text[i])

    contexts = np.array(contexts)
    targets = np.array(targets)

    num_samples = len(targets)
    self.num_samples = num_samples
    shuffled_idx = np.random.permutation(num_samples)

    for i in range(0, num_samples, batch_size):
      batch_indexes = shuffled_idx[i:i+batch_size]

      yield contexts[batch_indexes], targets[batch_indexes]

  def get_num_batches(self, window_size, batch_size):
      ray = (window_size-1)//2
      num_samples = len(self.tokenized_text) - 2*ray
      return (num_samples + batch_size - 1) // batch_size


def load_books(folder):
  texts = []

  for file in os.listdir(folder):
      if file.endswith(".txt"):
          with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
              raw = f.read()
              match = re.search(
                  r"\*\*\* START OF THIS PROJECT.*?\*\*\*(.*?)\*\*\* END OF THIS PROJECT.*?\*\*\*", 
                  raw, 
                  re.DOTALL
              )
              if match:
                  cleaned = match.group(1).strip()
                  texts.append(cleaned)

  return " ".join(texts)