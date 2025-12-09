import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
import argparse
import time
# from tensorflow.keras.preprocessing.sequence import pad_sequences # Use Keras function for easy padding

# Load the data (assuming you ran the initial loading/padding part from the Keras guide)
# If not, run these lines first:

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica_25', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
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
opt = vars(opt)
print(opt)

tra_seqs, tr_labs = pickle.load(open(f'{opt["dataset"]}/train.txt', 'rb'))
all_train_seq = pickle.load(open(f'{opt["dataset"]}/all_train_seq.txt', 'rb'))
te_seqs, te_labs = pickle.load(open(f'{opt["dataset"]}/test.txt', 'rb'))

# # --- DATA REDUCTION
# REDUCTION_PORTION = 0.05 
# print(f"Reducing data to {REDUCTION_PORTION*100}% to fast testing.")

# # --- REDUCCIÓN DEL CONJUNTO DE ENTRENAMIENTO ---
# N_TRAIN_SAMPLES = len(tra_seqs)
# n_train_subset = int(N_TRAIN_SAMPLES * REDUCTION_PORTION)

# # --- Get indices without replacement
# train_indices = np.random.choice(N_TRAIN_SAMPLES, size=n_train_subset, replace=False)

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

# print(len(X_train))
# print(X_train[:10])

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

# Define GNN Model

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

# Initialize the model
EMBED_DIM = 100
HIDDEN_UNITS = 128

model = GRUSessionModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_UNITS)
# Push the model to the GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

print(f"Model will run on: {DEVICE}")

# --- 4. FUNCIÓN REUTILIZABLE PARA EVALUACIÓN ---
def evaluate_model(model, data_loader, criterion, device):
  """
  Calcula la pérdida promedio, la precisión, y recopila las etiquetas
  verdaderas y predichas para métricas avanzadas.
  """
  model.eval() # Poner el modelo en modo evaluación
  total_loss = 0.0
  n_samples = 0
  
  # Listas para almacenar todas las etiquetas
  all_labels = []
  all_predicted = []
  
  with torch.no_grad():
    for seqs, labels in data_loader:
      seqs = seqs.to(device)
      labels = labels.to(device)
      
      outputs = model(seqs)
      loss = criterion(outputs, labels)
      total_loss += loss.item() * seqs.size(0)
      
      _, predicted = torch.max(outputs, 1)
      n_samples += labels.size(0)
      
      # --- Almacenamiento de Etiquetas ---
      all_labels.extend(labels.cpu().numpy())
      all_predicted.extend(predicted.cpu().numpy())
          
  avg_loss = total_loss / n_samples
  
  # Convertir a arrays de NumPy
  y_true = np.array(all_labels)
  y_pred = np.array(all_predicted)
  
  # Calcular Accuracy simple (para consistencia)
  accuracy = 100.0 * np.sum(y_true == y_pred) / n_samples
  
  model.train() # Devolver el modelo a modo entrenamiento
  
  # Retornar más métricas para el reporte avanzado
  return avg_loss, accuracy, y_true, y_pred

# -- Estructuras de datos para resultados --
history = {
  'train_loss': [],
  # 'val_loss': [],
  'train_acc': [],
  # 'val_acc': []
}

performance_metrics = {
  'total_training_time_s': 0.0,
  'epoch_times_s': [],
  'inference_time_s': 0.0,
  'predictions_per_second': 0.0,
}

def train_epoch(model, dataloader, criterion, optimizer, device):
  epoch_start_time = time.time()
  model.train()
  
  # Variables de Acumulación
  total_loss = 0.0
  total_correct = 0 # Para acumular las predicciones correctas
  n_train_samples = 0
  
  # --- Bucle de Entrenamiento ---
  for i, (seqs, labels) in enumerate(dataloader): 
    # Mover datos a dispositivo
    seqs, labels = seqs.to(device), labels.to(device)
    batch_size = seqs.size(0)
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(seqs) # outputs shape: (batch_size, VOCAB_SIZE)
    
    # Calculate loss
    loss = criterion(outputs, labels)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # --- Cálculo de Métrica de Entrenamiento (Precisión Top-1) ---
    
    # Obtener la predicción con la puntuación más alta (Top-1)
    # torch.max devuelve (valores, índices). Nos interesan los índices.
    _, predicted = torch.max(outputs.data, 1)
    
    # Contar cuántas predicciones coinciden con las etiquetas reales
    total_correct += (predicted == labels).sum().item()
    
    # Acumular pérdida total y número de muestras
    total_loss += loss.item() * batch_size
    n_train_samples += batch_size

    # --------------------------------------------------------------------------------
    # --- Puntos de Control (Reporte cada 100 lotes) ---
    # --------------------------------------------------------------------------------
    
    # if (i + 1) % 100 == 0: 
    #   # 1. Pérdida del Lote Actual
    #   current_batch_loss = loss.item()
      
    #   # 2. Precisión del Lote Actual
    #   current_batch_accuracy = (predicted == labels).sum().item() / batch_size
      
    #   # 3. Pérdida Promedio Acumulada (desde el inicio de la época)
    #   running_avg_loss = total_loss / n_train_samples
      
    #   # 4. Precisión Promedio Acumulada (desde el inicio de la época)
    #   running_avg_accuracy = total_correct / n_train_samples
      
    #   print(f"Batch [{i+1}/{len(dataloader)}] | "
    #         f"Loss: {current_batch_loss:.4f} (Avg: {running_avg_loss:.4f}) | "
    #         f"Acc: {current_batch_accuracy:.4f} (Avg: {running_avg_accuracy:.4f}) | "
    #         f"Time: {time.time() - epoch_start_time:.2f}s")
  
  # --- Resultados Finales de la Época ---
  epoch_duration = time.time() - epoch_start_time
  
  avg_train_loss = total_loss / n_train_samples
  avg_train_accuracy = total_correct / n_train_samples
  
  # Devolver las métricas clave para el reporte de la época
  return avg_train_loss, avg_train_accuracy, epoch_duration

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
# print("\n--- Starting PyTorch Model Training ---")

# for epoch in range(1, EPOCHS + 1):
#     avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
#     # Evaluate on the test set
#     test_acc, test_topk_acc = evaluate_model(model, test_loader, DEVICE, k=10)
    
#     print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | '
#           f'Test Acc: {test_acc:.4f} | Test Top-10 Acc: {test_topk_acc:.4f}')

import torch
import torch.nn.functional as F

import torch

def calculate_recommender_metrics(outputs, labels, k=20):
    """
    Calcula Recall@K (Hit Ratio) y MRR@K.
    
    Args:
        outputs (torch.Tensor): Puntuaciones logit del modelo (batch_size, VOCAB_SIZE).
        labels (torch.Tensor): Etiquetas reales (batch_size).
        k (int): El umbral K (20).

    Returns:
        tuple: (Recall_K (HR), MRR_K), ambos entre 0 y 1.
    """
    # 1. Obtener los K mejores índices (índices = predicciones de ítems)
    # top_k_indices.shape: (Batch, K)
    top_k_indices = torch.topk(outputs, k)[1]
    
    # 2. Preparar las etiquetas (rankings del target)
    # target_labels.shape: (Batch, 1)
    target_labels = labels.unsqueeze(-1)
    
    # 3. Calcular Hits (True/False si el target está en el Top K)
    # hits_k.shape: (Batch, K)
    hits_k = (top_k_indices == target_labels)
    
    # --- Recall@K (Hit Ratio) ---
    # El target está en el Top K si cualquier elemento en la fila es True.
    # .any(dim=1) devuelve True (1.0) o False (0.0) para cada muestra.
    # .float().mean().item() calcula el promedio sobre el lote (el Hit Ratio).
    recall_k = hits_k.any(dim=1).float().mean().item()

    # --- MRR@K (Mean Reciprocal Rank) ---
    mrr_sum = 0.0
    
    # Iterar sobre el lote para encontrar el rank exacto.
    for i in range(labels.size(0)):
        # hits_k[i] es la fila (K elementos) para esta muestra
        
        # Encuentra el índice (rank - 1) donde hits_k[i] es True
        # La función .nonzero(as_tuple=True)[0] devuelve los índices donde es True
        ranks = hits_k[i].nonzero(as_tuple=True)[0] 
        
        if ranks.numel() > 0:
            # El rank (posición) es el índice + 1. 
            # Si hay múltiples 'hits' (lo cual no debería pasar con un solo target), 
            # tomamos el primero (el mejor rank).
            rank = ranks[0].item() + 1 
            mrr_sum += 1.0 / rank
    
    mrr_k = mrr_sum / labels.size(0)

    return recall_k, mrr_k # Ambos entre 0 y 1

    model.eval() # Poner el modelo en modo evaluación
    
    total_precision_k = 0.0
    total_mrr_k = 0.0
    n_batches = 0
    
    with torch.no_grad(): # Desactivar el cálculo de gradientes
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            outputs = model(seqs)
            
            # Calcular métricas para el lote
            precision_k, mrr_k = calculate_recommender_metrics(outputs, labels, k=k)
            
            total_precision_k += precision_k
            total_mrr_k += mrr_k
            n_batches += 1
            
    # Calcular promedios generales
    avg_precision_k = total_precision_k / n_batches
    avg_mrr_k = total_mrr_k / n_batches
    
    return avg_precision_k, avg_mrr_k
def evaluate_model_recommender_metrics(model, dataloader, device, k=20):
    model.eval()
    total_recall_k = 0.0
    total_mrr_k = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            outputs = model(seqs)
            
            # Usar la nueva función ajustada
            recall_k, mrr_k = calculate_recommender_metrics(outputs, labels, k=k)
            
            total_recall_k += recall_k
            total_mrr_k += mrr_k
            n_batches += 1
            
    # Valores promedio (entre 0 y 1)
    avg_recall_k = total_recall_k / n_batches
    avg_mrr_k = total_mrr_k / n_batches
    
    # Devolver los valores multiplicados por 100 para coincidir con el código de referencia
    return avg_recall_k * 100, avg_mrr_k * 100

BATCH_SIZE = opt["batchSize"]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")

EPOCHS = 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=opt["l2"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_dc_step"], gamma=opt["lr_dc"])

EPOCHS = opt["epoch"]
K_VALUE = 20

print(f"\n--- Starting Training ###")

total_training_time_start = time.time()

for epoch in range(1, EPOCHS + 1):
    avg_train_loss, avg_train_accuracy, epoch_duration = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    history["train_loss"] = avg_train_loss
    history["train_acc"] = avg_train_accuracy
    
    # Evaluate on the test set using the new function
    test_recall_k, test_mrr_k = evaluate_model_recommender_metrics(model, test_loader, DEVICE, k=K_VALUE)

    scheduler.step()
    
    print(f'Epoch {epoch}/{EPOCHS} | '
          f'Train Loss: {avg_train_loss:.4f} | '
          f'Train Accuracy: {avg_train_accuracy: .4f} | '
          f'Test Recall@{K_VALUE}: {test_recall_k:.4f} | ' # <--- NUEVA MÉTRICA
          f'Test MRR@{K_VALUE}: {test_mrr_k:.4f}') # <--- NUEVA MÉTRICA

  # --- CÁLCULO DEL TIEMPO TOTAL DE ENTRENAMIENTO ---
total_training_duration = time.time() - total_training_time_start
print("Total training time: ", total_training_duration)
performance_metrics['total_training_time_s'] = total_training_duration # 1. Guardar el tiempo total
print("\n--- Training Complete ---")
