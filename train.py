import numpy as np
from tqdm import tqdm
from model import Word2Vec
from dataloader import DataLoader, load_books


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


text = load_books("books")
size = 1_000_000
text = text[:size] 
train_size = int(0.8 * size)
train_set = text[:train_size]
test_set = text[train_size:]


dataloader = DataLoader(text)
dictionary = dataloader.dictionary
dict_size = len(dictionary)


train_dataloader = DataLoader(train_set, dictionary)
test_dataloader = DataLoader(test_set, dictionary)

reversed_dictionary = dataloader.reverse_dictionary
model = Word2Vec(dict_size, reversed_dictionary=reversed_dictionary, embedding_size=50, window_size=5, lr=0.1)

train(model, train_dataloader, test_dataloader, batch_size=32)