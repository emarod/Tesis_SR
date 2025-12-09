import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import random

REDUCTION_FACTOR = 1 # Keep 25% of the data
NEW_DATASET_FOLDER = 'diginetica_weights'
RANDOM_SEED = 42

random.seed(RANDOM_SEED) 
print(f"Set random seed to: {RANDOM_SEED}")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

data_path = "../data/dataset-train-diginetica/"

if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'

print("-- Starting @ %ss" % datetime.datetime.now())
# (Code for reading data, building sess_clicks and sess_date, and initial filtering)
# ... (Your original code remains unchanged up to this point) ...
with open(data_path + dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    # ... (Rest of data reading and session sorting) ...
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        item = data['itemId'], int(data['timeframe'])
        curdate = data['eventdate']
        
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1

    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_timeframe_delta = {}
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_timeframe_delta[i] = [c[1] for c in sorted_clicks]
    sess_date[curid] = date

# print(sess_timeframe_delta)
print("-- Reading data @ %ss" % datetime.datetime.now())
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        del sess_timeframe_delta[s]

with open('sess_clicks.txt', 'w') as f:
    f.write(str(sess_clicks))

with open('sess_date.txt', 'w') as f:
    f.write(str(sess_date))

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

# Remove items that appear less than 5 times
length = len(sess_clicks)
print("Before filtering", len(sess_clicks))
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    curtime = sess_timeframe_delta[s]
    
    filseq = []
    filtime = []

    # 1. Iterar sobre los índices de la secuencia
    for i in range(len(curseq)):
      item_id = curseq[i]
      
      # 2. Aplicar la condición (El item debe aparecer 5 o más veces)
      if iid_counts[item_id] >= 5:
          # Si el item pasa el filtro, lo guardamos
          filseq.append(item_id)
          # Y OBLIGATORIAMENTE guardamos su tiempo asociado
          filtime.append(curtime[i])

    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
        del sess_timeframe_delta[s]
    else:
        sess_clicks[s] = filseq
        sess_timeframe_delta[s] = filtime
print("After filtering", len(sess_clicks))


# Reduce the total number of sessions to 25%

original_sess_ids = list(sess_clicks.keys())
n_sessions_to_keep = int(len(original_sess_ids) * REDUCTION_FACTOR)

# Randomly sample the session IDs to keep
sampled_sess_ids = random.sample(original_sess_ids, n_sessions_to_keep)

# Create the reduced session dictionaries
reduced_sess_clicks = {sid: sess_clicks[sid] for sid in sampled_sess_ids}
reduced_sess_time = {sid: sess_timeframe_delta[sid] for sid in sampled_sess_ids}
reduced_sess_date = {sid: sess_date[sid] for sid in sampled_sess_ids}

# Replace the full dictionaries with the reduced ones
sess_clicks = reduced_sess_clicks
sess_timeframe_delta = reduced_sess_time
sess_date = reduced_sess_date

print(f"Total sessions before reduction: {len(original_sess_ids)}")
print(f"Total sessions after {REDUCTION_FACTOR*100}% reduction: {len(sess_clicks)}")


# # # Split out test set based on dates (uses the reduced sess_date)
dates = list(sess_date.items())
maxdate = dates[0][1]

#compute maxdate
for _, date in dates:
    if maxdate < date:
        maxdate = date

# # 7 days for test
splitdate = maxdate - (86400 * 7)

print('Splitting date', splitdate)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# # # Sort sessions by date (timeframe)
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# COMPUTE timeframe_delta
print(len(sess_clicks))
print(len(sess_timeframe_delta))
# print(sess_date)

##################################################### HERE #####################################

# # Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# # Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    # ... (Your original obtian_tra logic remains, but runs on the reduced tra_sess) ...
    train_ids = []
    train_seqs = []
    train_times = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_times += [sess_timeframe_delta[s]]
    return train_ids, train_dates, train_seqs, train_times

# # # Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    # ... (Your original obtian_tes logic remains, but runs on the reduced tes_sess 
    # and uses the smaller item_dict) ...
    test_ids = []
    test_seqs = []
    test_times = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            # THIS IS THE CRITICAL FILTERING STEP!
            # It ensures the test set only uses items that are in the new, smaller item_dict.
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_times += [sess_timeframe_delta[s]]
    return test_ids, test_dates, test_seqs, test_times

tra_ids, tra_dates, tra_seqs, tra_times = obtian_tra()
tes_ids, tes_dates, tes_seqs, tes_times = obtian_tes()

print(tra_seqs[:10])
print(tra_times[:10])

# Data augmentation
def process_seqs(iseqs, idates, itimes):
  # ... (Your original process_seqs logic remains) ...
  out_seqs = []
  out_dates = []
  out_times = []
  labs = []
  ids = []
  # print(len(iseqs))
  # print(len(itimes))
  for id, seq, date, tim in zip(range(len(iseqs)), iseqs, idates, itimes):
    for i in range(1, len(seq)):
      tar = seq[-i]
      labs += [tar]
      out_seqs += [seq[:-i]]
      out_dates += [date]
      out_times += [tim[:-i]]
      ids += [id]
  return out_seqs, out_dates, out_times, labs, ids

tr_seqs, tr_dates, tr_times, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, tra_times)
print("#####TR IDS: ", len(set(tr_ids)))
te_seqs, te_dates, te_times, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, tes_times)
print(len(tr_seqs))
print(len(tr_times))
print(tr_seqs[:6])
print(tr_times[:6])

import math
print("Normalization of times")

import math
import copy

def calculate_global_mean_dwell_time(session_times_original):
    """
    PASO 1: Calcula la media global (mu_g) sobre los tiempos BRUTOS.
    """
    total_duration_sum = 0.0
    total_transitions = 0
    
    # Usamos una copia para no alterar las listas originales
    sessions = copy.deepcopy(session_times_original)
    
    for session in sessions:
        if len(session) > 1:
            for idx in range(1, len(session)):
                # Dwell Time bruto
                delta_t = session[idx] - session[idx-1]
                
                # Manejo de casos no válidos (asumiendo que quieres contarlos)
                if delta_t <= 0:
                    delta_t = 1e-6 

                total_duration_sum += delta_t
                total_transitions += 1
                
    if total_transitions > 0:
        mu_g = total_duration_sum / total_transitions
    else:
        mu_g = 1.0 
        
    return mu_g

def process_times(session_times_original, mu_g):
    """
    PASO 2: Calcula el Dwell Time, normaliza e imputa con mu_g_norm.
    """
    output = []
    
    # 1. Normalización de la Media Global (mu_g)
    mu_g_norm = math.log(mu_g + 1)
    
    for session in session_times_original:
        new_session = []
        
        # Caso A: Sesión de longitud 1
        if len(session) == 1:
            new_session.append(mu_g_norm)
            
        # Caso B: Sesión de longitud > 1
        else:
            for idx in range(1, len(session)):
                # Dwell Time BRUTO
                dwell_time = session[idx] - session[idx-1]
                
                if dwell_time <= 0:
                    dwell_time = 1e-6
                
                # Normalizar y añadir
                dwell_time_norm = math.log(dwell_time + 1)
                new_session.append(dwell_time_norm)
                
            # Imputar el último ítem
            new_session.append(mu_g_norm)
            
        output.append(new_session)
        
    return output


global_mean = calculate_global_mean_dwell_time(tr_times)
tr_times = process_times(tr_times, global_mean)

tra = (tr_seqs, tr_labs, tr_times)
tes = (te_seqs, te_labs, te_times)

all_len = 0
for seq in tra_seqs:
    all_len += len(seq)
for seq in tes_seqs:
    all_len += len(seq)
print('avg length: ', all_len/(len(tra_seqs) + len(tes_seqs) * 1.0))

# --- FINAL STEP: Save to the new, reduced directory ---
if not os.path.exists(NEW_DATASET_FOLDER):
    os.makedirs(NEW_DATASET_FOLDER)

def save_as_human_readable(data_list, filename):
    """
    Guarda una lista de secuencias (o listas) en un archivo de texto, 
    donde cada secuencia se convierte en una línea separada y los elementos 
    dentro de la secuencia se unen por un espacio.
    """
    with open(filename, 'w') as f:
        for sequence in data_list:
            # --- CORRECCIÓN ---
            # Verifica si el elemento no es iterable (ej. es un int o string simple)
            # y si no es un string (para evitar tratar strings como secuencias de caracteres).
            if not isinstance(sequence, (list, tuple)) or isinstance(sequence, str):
                # Si no es una lista/tupla (y no es un string), lo envuelve en una lista.
                # Ejemplo: 4 se convierte a [4]
                sequence = [sequence]
            # ------------------
            
            # Asegura que todos los elementos sean strings antes de unirlos
            line = ' '.join(map(str, sequence))
            f.write(line + '\n')
    print(f"Guardado exitosamente como texto legible en: {filename}")

save_as_human_readable(tra, f'{NEW_DATASET_FOLDER}/train_txt.txt')
save_as_human_readable(tes, f'{NEW_DATASET_FOLDER}/test_txt.txt')
save_as_human_readable(seq, f'{NEW_DATASET_FOLDER}/all_train_seq_txt.txt')

    
pickle.dump(tra, open(f'{NEW_DATASET_FOLDER}/train.txt', 'wb'))
pickle.dump(tes, open(f'{NEW_DATASET_FOLDER}/test.txt', 'wb'))
pickle.dump(tra_seqs, open(f'{NEW_DATASET_FOLDER}/all_train_seq.txt', 'wb'))

print(f'\nDone. New reduced dataset saved in the "{NEW_DATASET_FOLDER}" folder.')