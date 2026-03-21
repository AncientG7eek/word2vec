import numpy as np

class Word2Vec:
  def __init__(self, dict_size, embedding_size, window_size, reversed_dictionary, lr=0.001):

    self.window_size = window_size
    self.reversed_dictionary = reversed_dictionary
    self.lr = lr
    self.lookup_matrix = np.random.uniform(0,1,(dict_size, embedding_size))
    self.logit_matrix = np.random.uniform(0,1,(dict_size, embedding_size))

  def forward(self, context_indexes_batch, target_index_batch): # batch_size,window_size-1

    context_vectors = self.lookup_matrix[context_indexes_batch] # batch_size,context_size,embedding_size
    avg_context_vector = context_vectors.mean(axis=1) # batch_size,embedding_size
    logits = avg_context_vector @ self.logit_matrix.T # batch_size,embedding_size @ embedding_size,dict_size = batch_size,dict_size
    probs = self.softmax(logits) # batch_size,dict_size

    return probs, logits, avg_context_vector

  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

  def backwards(self, context_indexes_batch, target_index_batch):
    probs, logits, avg_context_vector = self.forward(context_indexes_batch, target_index_batch)
    batch_size, context_size = context_indexes_batch.shape
    
    # gradient wrt logits
    dL_dz = probs.copy()
    dL_dz[np.arange(batch_size),target_index_batch] -= 1
    dL_dz /= batch_size

    # gradient wrt logit_matrix
    dL_dW = dL_dz.T @ avg_context_vector

    # gradient wrt avg_context_vetor
    dL_da = dL_dz @ self.logit_matrix

    # distribute gradient to all words from the context
    dL_dc = dL_da[:,None,:] / context_size

    self.learning_step(context_indexes_batch, dL_dW, dL_dc)


  def learning_step(self, context_indexes_batch, logit_matrix_gradient, lookup_matrix_gradient, lr=0.001):
    self.logit_matrix -= self.lr * logit_matrix_gradient
    np.add.at(self.lookup_matrix, context_indexes_batch, -self.lr * lookup_matrix_gradient)


  def softmax_prime(self, z):
    return self.softmax(z) * (1 - self.softmax(z))

  def loss(self, probs, target_index_batch): # averaged context vector vs target word vetor
    batch_size = np.arange(len(target_index_batch))
    return -np.mean(np.log(probs[batch_size, target_index_batch]))

  def evaluate(self, probs, target_index_batch):
    batch_size = len(target_index_batch)
    top_k = np.argsort(probs, axis=1)[:, -5:]
    correct = sum(target in top_k[i] for i, target in enumerate(target_index_batch)) 
    return correct / batch_size




