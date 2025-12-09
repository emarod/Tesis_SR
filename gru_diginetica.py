import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences # Use Keras function for easy padding
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
# parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

# Load the data (assuming you ran the initial loading/padding part from the Keras guide)
# If not, run these lines first:
tra_seqs, tr_labs = pickle.load(open('diginetica/train.txt', 'rb'))
all_train_seq = pickle.load(open('diginetica/all_train_seq.txt', 'rb'))
te_seqs, te_labs = pickle.load(open('diginetica/test.txt', 'rb'))

print("Trainning sequences len: ", len(tra_seqs))
print("Trainning labels len: ", len(tr_labs))
print("All training sequences len: ", len(all_train_seq))
print("Test sequences len: ", len(te_seqs))
print("Test labels len: ", len(te_labs))

# Determine VOCAB_SIZE (MAX ITEM ID)
max_item_index = 0
for seq in all_train_seq:
    max_item_index = max(max_item_index, max(seq))
VOCAB_SIZE = max_item_index + 1

# # Determine MAX_SEQ_LEN
MAX_SEQ_LEN = max(len(s) for s in tra_seqs) 

# Pad Sequences
X_train = pad_sequences(tra_seqs, maxlen=MAX_SEQ_LEN, padding='pre', value=0)
X_test = pad_sequences(te_seqs, maxlen=MAX_SEQ_LEN, padding='pre', value=0)
y_train = np.array(tr_labs)
y_test = np.array(te_labs)

print(len(X_train))
print(X_train[:10])

class SequentialDataset(Dataset):
    def __init__(self, sequences, labels):
        # Convert numpy arrays to PyTorch Tensors
        # Sequences are longs because they are indices for the embedding layer
        self.sequences = torch.LongTensor(sequences)
        # Labels are longs because they are target indices for the loss function
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create Dataset and DataLoader instances
train_dataset = SequentialDataset(X_train, y_train)
test_dataset = SequentialDataset(X_test, y_test)

BATCH_SIZE = 512

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")

# Define GNRNNN Model

class GRUSessionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units):
        super(GRUSessionModel, self).__init__()
        
        self.hidden_units = hidden_units
        
        # 1. Embedding Layer
        # padding_idx=0: PyTorch handles padding by mapping index 0 to a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. GRU Layer
        # batch_first=True: Expects input shape (batch_size, seq_len, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_units, batch_first=True)
        
        # 3. Output Layer (Linear/Dense)
        # Input size is the GRU hidden size, output size is the vocabulary size
        self.fc = nn.Linear(hidden_units, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        
        # 2. GRU
        # output is the sequence of hidden states for every step: (batch_size, seq_len, hidden_units)
        # hidden is the final hidden state: (1, batch_size, hidden_units)
        output, hidden = self.gru(embedded)
        
        # 3. Select the output corresponding to the last item in the sequence
        # We use the final hidden state 'hidden' (after squeezing dimension 0), 
        # as it summarizes the entire sequence up to the last element.
        final_state = hidden.squeeze(0)
        
        # 4. Output Layer
        logits = self.fc(final_state)
        # logits shape: (batch_size, vocab_size)
        
        return logits

import torch.nn as nn

class SimpleRNNSessionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units):
        super(SimpleRNNSessionModel, self).__init__()
        
        self.hidden_units = hidden_units
        
        # 1. Embedding Layer (Identical to GRU)
        # padding_idx=0 is essential for masking the padded values
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Simple RNN Layer (The key difference)
        # We replace nn.GRU with nn.RNN
        # batch_first=True: Expects input shape (batch_size, seq_len, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_units, batch_first=True)
        
        # 3. Output Layer (Identical to GRU)
        self.fc = nn.Linear(hidden_units, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embed_dim)
        
        # 2. RNN Forward Pass
        # output: sequence of hidden states for every step: (batch_size, seq_len, hidden_units)
        # hidden: final hidden state: (1, batch_size, hidden_units)
        output, hidden = self.rnn(embedded)
        
        # 3. Select the final hidden state to predict the next item
        final_state = hidden.squeeze(0)
        
        # 4. Output Layer
        logits = self.fc(final_state)
        # logits shape: (batch_size, vocab_size)
        
        return logits

# Initialize the model
EMBED_DIM = 100
HIDDEN_UNITS = 128

model = GRUSessionModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_UNITS)
# Push the model to the GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

print(f"Model will run on: {DEVICE}")

import torch.optim as optim

# Trainning Parameters

# Loss: CrossEntropyLoss is the standard for multi-class classification
# It combines LogSoftmax and Negative Log Likelihood Loss.
# Importantly, it takes integer labels, similar to Keras's sparse_categorical_crossentropy.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 1

# Training Loop

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    # --- CHANGE: Use enumerate to get both the index (i) and the data (seqs, labels) ---
    for i, (seqs, labels) in enumerate(dataloader): 
        # Move data to device
        seqs, labels = seqs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(seqs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # We accumulate the loss for the whole epoch here
        total_loss += loss.item() * seqs.size(0) # Multiply by batch size for correct average later

        # --------------------------------------------------------------------------------
        # --- Print Info Here ---
        # --------------------------------------------------------------------------------
        
        # Checkpoint: Print every N batches (e.g., every 100 batches)
        if (i + 1) % 100 == 0: 
            # 1. Calculate the current batch's average loss
            current_batch_loss = loss.item()
            
            # 2. Calculate the average loss accumulated so far in the epoch
            # (i+1) is the number of batches processed so far
            running_avg_loss = total_loss / ((i + 1) * seqs.size(0))
            
            print(f"    Batch [{i+1}/{len(dataloader)}] | "
                  f"Batch Loss: {current_batch_loss:.4f} | "
                  f"Running Avg Loss: {running_avg_loss:.4f} | "
                  f"Seq Batch Size: {seqs.size(0)}")
        
    # Corrected return: total_loss divided by the total number of samples
    # We used seqs.size(0) to get the batch size, so we need the total number of samples.
    total_samples = len(dataloader.dataset)
    return total_loss / total_samples

# Top 20 accuracy
# def evaluate_model(model, dataloader, device, k=10):
#     model.eval()
#     total_correct = 0
#     total_at_k = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for seqs, labels in dataloader:
#             seqs, labels = seqs.to(device), labels.to(device)
            
#             outputs = model(seqs)
            
#             # Accuracy (Top-1)
#             _, predicted = torch.max(outputs.data, 1)
#             total_correct += (predicted == labels).sum().item()
            
#             # Top-K Accuracy
#             # torch.topk gets the top k values and their indices
#             _, topk_indices = torch.topk(outputs, k=k) 
            
#             # labels need to be reshaped to (batch_size, 1) to be compared
#             labels_reshaped = labels.view(-1, 1)
            
#             # Check if the true label is in the top k predicted indices
#             in_topk = torch.any(topk_indices == labels_reshaped, dim=1)
#             total_at_k += in_topk.sum().item()
            
#             total_samples += labels.size(0)
            
#     accuracy = total_correct / total_samples
#     top_k_accuracy = total_at_k / total_samples
    
#     return accuracy, top_k_accuracy

# --- Start the full training loop ---
print("\n--- Starting PyTorch Model Training ---")

# for epoch in range(1, EPOCHS + 1):
#     avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
#     # Evaluate on the test set
#     test_acc, test_topk_acc = evaluate_model(model, test_loader, DEVICE, k=10)
    
#     print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | '
#           f'Test Acc: {test_acc:.4f} | Test Top-10 Acc: {test_topk_acc:.4f}')

def evaluate_model_recommender_metrics(model, dataloader, device, k=20):
    model.eval()
    
    total_precision = 0.0
    total_mrr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            # Forward pass to get scores (logits)
            outputs = model(seqs)
            
            # Get the indices (item IDs) of the top K predicted items
            # dim=1 means we look along the item dimension (vocab_size)
            # largest=True ensures we get the highest scores
            _, topk_indices = torch.topk(outputs, k=k, dim=1, largest=True)
            # topk_indices shape: (batch_size, k)
            
            # --- Calculate Precision@K (P@20) ---
            # labels are the true next item indices (batch_size,)
            # Reshape labels to (batch_size, 1) to enable comparison with topk_indices
            labels_reshaped = labels.view(-1, 1)
            
            # Check if the true label is present anywhere in the top K predictions
            # in_topk shape: (batch_size,) (True/False for each sample)
            in_topk = torch.any(topk_indices == labels_reshaped, dim=1)
            
            # Precision@K is the percentage of samples where the correct item was in the top K
            # We average the boolean tensor (True=1, False=0)
            precision_k = in_topk.float().mean()
            total_precision += precision_k.item()
            
            # --- Calculate Mean Reciprocal Rank@K (MRR@20) ---
            
            # Find the rank of the true label within the top K predictions
            # ranks shape: (batch_size, k)
            ranks = (topk_indices == labels_reshaped).nonzero(as_tuple=True)
            
            # ranks[0] is the row index (sample index in the batch)
            # ranks[1] is the column index (rank within the top K, starting from 0)
            
            if len(ranks[0]) > 0:
                # The rank is ranks[1] + 1 (since rank starts at 1)
                # The reciprocal rank is 1 / (rank)
                reciprocal_ranks = 1.0 / (ranks[1].float() + 1.0)
                
                # To handle multiple matches per sample (which shouldn't happen with unique item IDs), 
                # we only take the first match (the highest rank).
                # Since we are iterating over all samples, we need to map the reciprocal rank 
                # back to the original batch index (ranks[0]) and sum them up.
                
                # Create an empty tensor for reciprocal ranks for the entire batch
                batch_mrr = torch.zeros(labels.size(0), dtype=torch.float, device=device)
                
                # For each sample index that had a hit, assign its reciprocal rank
                batch_mrr[ranks[0]] = reciprocal_ranks
                
                total_mrr += batch_mrr.sum().item()
            
            total_samples += labels.size(0)

    # Final calculation is the average across all batches
    final_p_k = total_precision / len(dataloader)
    final_mrr_k = total_mrr / total_samples
    
    return final_p_k, final_mrr_k

# Assuming your model, dataloaders, and DEVICE are defined...
K_VALUE = 20
EPOCHS = 10

print(f"\n--- Starting Training with P@{K_VALUE} and MRR@{K_VALUE} Evaluation ---")

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Evaluate on the test set using the new function
    test_p_k, test_mrr_k = evaluate_model_recommender_metrics(model, test_loader, DEVICE, k=K_VALUE)
    
    print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | '
          f'Test P@{K_VALUE}: {test_p_k:.4f} | Test MRR@{K_VALUE}: {test_mrr_k:.4f}')

print("\n--- Training Complete ---")
