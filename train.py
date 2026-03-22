import numpy as np
from tqdm import tqdm
from model import Word2Vec
from dataloader import DataLoader, Config, load_books
import yaml

with open('configs/config.yaml', "r") as f:
    config = yaml.safe_load(f)

config = Config(config)

def train(model, train_dataloader, test_dataloader, epochs=10, batch_size=16):
  for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_iterator = train_dataloader.load(model.window_size, batch_size)
    train_steps = train_dataloader.get_num_batches(model.window_size, batch_size)
    train_bar = tqdm(train_iterator, total=train_steps, desc="Training", leave=False)

    test_iterator = test_dataloader.load(model.window_size, batch_size)
    test_steps = test_dataloader.get_num_batches(model.window_size, batch_size)
    test_bar = tqdm(test_iterator, total=test_steps, desc="Testing", leave=False)
    
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for contexts, targets in train_bar:
      probs,*_= model.forward(contexts,targets)
      loss = model.loss(probs,targets)
      train_losses.append(loss)
      acc = model.evaluate(probs, targets)
      #print(f"accuracy: {acc}")
      train_accuracies.append(acc)

      model.backwards(contexts,targets)
    train_accuracy = np.mean(train_accuracies)
    print(f"Training accuracy: {train_accuracy}")
    print(f"Training loss: {np.mean(train_losses)}")

    for contexts, targets in test_bar:
      probs,*_= model.forward(contexts,targets)
      loss = model.loss(probs,targets)
      test_losses.append(loss)
      acc = model.evaluate(probs, targets)
      #print(f"accuracy: {acc}")
      test_accuracies.append(acc)

    test_accuracy = np.mean(test_accuracies)
    print(f"Testing accuracy: {test_accuracy}")
    print(f"Testining loss: {np.mean(test_losses)}")


text = load_books(config.data.data_path)
size = config.data.size
text = text[:size] 
train_size = int(config.data.train_fraction * size)
train_set = text[:train_size]
test_set = text[train_size:]


dataloader = DataLoader(text)
dictionary = dataloader.dictionary
dict_size = len(dictionary)


train_dataloader = DataLoader(train_set, dictionary)
test_dataloader = DataLoader(test_set, dictionary)

reversed_dictionary = dataloader.reverse_dictionary
embedding_size=config.model.embedding_size
window_size=config.model.window_size
lr=config.model.learning_rate
model = Word2Vec(dict_size, reversed_dictionary=reversed_dictionary, embedding_size=embedding_size, window_size=window_size, lr=lr)

batch_size=config.training.batch_size
epochs=config.training.epochs
train(model, train_dataloader, test_dataloader, batch_size=batch_size, epochs=epochs)

np.save("lookup_matrix.npy", model.lookup_matrix)
np.save("logit_matrix.npy", model.logit_matrix)