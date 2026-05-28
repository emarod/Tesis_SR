import time
import csv
import pickle
import operator
import datetime
import os
import random
import argparse
import pandas as pd
import numpy as np

REDUCTION_FACTOR = 1 # Keep 25% of the data
NEW_DATASET_FOLDER = 'yoochoose_original_sessions'
RANDOM_SEED = 42

random.seed(RANDOM_SEED) 
print(f"Set random seed to: {RANDOM_SEED}")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

data_path = "../../../data/dataset-yoochoose/"

if opt.dataset == 'yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())

optimized_dtypes = {
    # Usar el tipo más pequeño posible si los IDs son < 32767 (int16)
    "sessionId": 'int32',
    "itemId": 'int32',
    "category": 'string' # Ajusta si tienes más de 127 categorías
}

# --- 1. Lectura y Conversión Vectorizada ---
print("Cargando y preprocesando datos...")
data = pd.read_csv(
    filepath_or_buffer=data_path + dataset,
    delimiter=",",
    dtype=optimized_dtypes,
)

# A. Conversión a Datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# B. Conversión Rápida a Unix Timestamp (segundos)
# Usa el método vectorizado más rápido para datetime64[ns]
data['unix_time'] = (data['timestamp'].values.astype(np.int64) // 10**9)

# Liberar la columna 'timestamp' de la memoria
del data['timestamp'] 

# --- 2. Agrupación y Conversión a Diccionarios (`to_dict()`) ---
# Esta es la parte más lenta (agg(list)), pero es la forma más rápida de obtener 
# listas de secuencias por grupo en Pandas.

print("Agrupando y convirtiendo a diccionarios...")

# sess_clicks: Lista de itemId por sessionId
sess_clicks = data.groupby('sessionId')['itemId'].agg(list).to_dict()

# sess_times: Lista de unix_time por sessionId
sess_times = data.groupby('sessionId')['unix_time'].agg(list).to_dict()

# sess_date: Última marca de tiempo (max) por sessionId
# El código original calcula la fecha del último evento para la sesión.
sess_date = data.groupby('sessionId')['unix_time'].max().to_dict()

print("sess_clicks, sess_times y sess_date generados")
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_times[s]
        del sess_date[s]

with open('sess_clicks.txt', 'w') as f:
  f.write(str(sess_clicks))

with open('sess_date.txt', 'w') as f:
  f.write(str(sess_date))
# print(sess_clicks)

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

# sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
print("Before filtering", len(sess_clicks))
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    curtime = sess_times[s]
    
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
        del sess_times[s]
    else:
        sess_clicks[s] = filseq
        sess_times[s] = filtime
print("After filtering", len(sess_clicks))

print("Session clicks len:", len(sess_clicks))
print("Session date len:", len(sess_date))

# Reduce the total number of sessions to 25%

original_sess_ids = list(sess_clicks.keys())
n_sessions_to_keep = int(len(original_sess_ids) * REDUCTION_FACTOR)

# Randomly sample the session IDs to keep
sampled_sess_ids = random.sample(original_sess_ids, n_sessions_to_keep)

# Create the reduced session dictionaries
reduced_sess_clicks = {sid: sess_clicks[sid] for sid in sampled_sess_ids}
reduced_sess_date = {sid: sess_date[sid] for sid in sampled_sess_ids}
reduced_sess_times = {sid: sess_times[sid] for sid in sampled_sess_ids}

# Replace the full dictionaries with the reduced ones
sess_clicks = reduced_sess_clicks
sess_date = reduced_sess_date
sess_times = reduced_sess_times

print(f"Total sessions before reduction: {len(original_sess_ids)}")
print(f"Total sessions after {REDUCTION_FACTOR*100}% reduction: {len(sess_clicks)}")

# # Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

#compute maxdate
for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    train_times = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        out_times = []
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
        train_times += [sess_times[s]]
    print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs, train_times


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    test_times = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_times += [sess_times[s]]
    return test_ids, test_dates, test_seqs, test_times


tra_ids, tra_dates, tra_seqs, tra_times = obtian_tra()
tes_ids, tes_dates, tes_seqs, tes_times = obtian_tes()

print("INFO ######")
print(len(tra_times))
print(len(tra_seqs))
print(tra_times[:3])
print(tra_seqs[:3])

# Data augmentation
def process_seqs(iseqs, idates, itimes):
    out_seqs = []
    out_dates = []
    out_times = []
    labs = []
    ids = []
    for id, seq, date, times in zip(range(len(iseqs)), iseqs, idates, itimes):
        labs += [seq[-1]]
        out_seqs += [seq[:-1]]
        out_dates += [date]
        out_times += [times[:-1]]
        ids += [id]
    return out_seqs, out_dates, out_times, labs, ids


tr_seqs, tr_dates, tr_times, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, tra_times)
te_seqs, te_dates, te_times, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, tes_times)

tra = (tr_seqs, tr_labs, tr_times)
tes = (te_seqs, te_labs, te_times)

all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)

tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:], tr_times[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:], tr_times[-split64:])
seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

print(len(tr_seqs[-split64:]))
print(len(tr_labs[-split64:]))
print(len(tr_times[-split64:]))

# --- FINAL STEP: Save to the new, reduced directory ---
if not os.path.exists(NEW_DATASET_FOLDER):
    os.makedirs(NEW_DATASET_FOLDER)

def save_as_txt(data_list, filename):
    """
    Guarda una lista de secuencias (o listas) en un archivo de texto, 
    donde cada secuencia se convierte en una línea separada y los elementos 
    dentro de la secuencia se unen por un espacio.
    """
    with open(filename, 'w') as f:
        for sequence in data_list:
            # Asegura que todos los elementos sean strings antes de unirlos
            line = ' '.join(map(str, sequence))
            f.write(line + '\n')
    print(f"Guardado exitosamente como texto legible en: {filename}")


save_as_txt(tra64, f'{NEW_DATASET_FOLDER}/train_txt.txt')
save_as_txt(tes, f'{NEW_DATASET_FOLDER}/test_txt.txt')
save_as_txt(seq64, f'{NEW_DATASET_FOLDER}/all_train_seq_txt.txt')
    
pickle.dump(tra64, open(f'{NEW_DATASET_FOLDER}/train.txt', 'wb'))
pickle.dump(tes, open(f'{NEW_DATASET_FOLDER}/test.txt', 'wb'))
pickle.dump(seq64, open(f'{NEW_DATASET_FOLDER}/all_train_seq.txt', 'wb'))

print(f'\nDone. New reduced dataset saved in the "{NEW_DATASET_FOLDER}" folder.')

print('Done.')