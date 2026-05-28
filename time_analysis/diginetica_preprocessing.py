import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import random
import pandas as pd
import math
import numpy as np
import copy

REDUCTION_FACTOR = 1 # Keep 25% of the data (1 = 100%, 0.25 = 25%)
NEW_DATASET_FOLDER = 'diginetica_original_sessions'
RANDOM_SEED = 42

random.seed(RANDOM_SEED) 
print(f"Set random seed to: {RANDOM_SEED}")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
opt, _ = parser.parse_known_args() # Ajustado para evitar crasheos en entornos interactivos/Jupyter
print(opt)

data_path = "../../../data/dataset-train-diginetica/"

if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'

print("-- Starting @ %ss" % datetime.datetime.now())
# --- 0. Definición de Tipos de Datos (Ajustado para Diginetica) ---
optimized_dtypes = {
    "sessionId": 'int32', 
    "itemId": 'int32',
    "timeframe": 'int32', 
    "eventdate": 'string' 
}

# --- 1. Lectura, Conversión y Ordenamiento ---
print("Cargando y preprocesando datos de Diginetica...")

data = pd.read_csv(
    filepath_or_buffer=data_path + dataset,
    delimiter=";", 
    dtype=optimized_dtypes,
    usecols=['sessionId', 'itemId', 'timeframe', 'eventdate'] 
)

# A. Conversión combinada a Unix Timestamp Absoluto (en segundos flotantes)
# Pasamos la fecha base del día a segundos Unix
unix_base_seconds = pd.to_datetime(data['eventdate']).values.astype(np.int64) // 10**9
# Pasamos el timeframe relativo de milisegundos a segundos
timeframe_seconds = data['timeframe'] / 1000.0

# Creamos la columna de tiempo unificada en segundos absolutos reales
data['unix_absolute_time'] = unix_base_seconds + timeframe_seconds

# B. Liberar la columna 'eventdate'
del data['eventdate'] 

# C. ORDENAMIENTO CRÍTICO
# Asegura que las filas estén perfectamente ordenadas de forma cronológica interna
data = data.sort_values(by=['sessionId', 'timeframe'])

# --- 2. Agrupación y Conversión a Diccionarios ---
print("Agrupando y convirtiendo a diccionarios...")

# sess_clicks: Lista de itemId por sessionId (ordenados)
sess_clicks = data.groupby('sessionId')['itemId'].agg(list).to_dict()

# sess_times: Lista de marcas de tiempo en segundos absolutos Unix por sessionId
sess_times = data.groupby('sessionId')['unix_absolute_time'].agg(list).to_dict()

# sess_date: Última fecha Unix (max) por sessionId para la partición temporal
sess_date = data.groupby('sessionId')['unix_absolute_time'].max().to_dict()

# Borrar la columna auxiliar
del data['unix_absolute_time']

print("sess_clicks, sess_times y sess_date generados")
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        del sess_times[s]

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
print("Before filtering", len(sess_clicks))
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    curtime = sess_times[s]
    
    filseq = []
    filtime = []

    for i in range(len(curseq)):
        item_id = curseq[i]
        if iid_counts[item_id] >= 5:
            filseq.append(item_id)
            filtime.append(curtime[i])

    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
        del sess_times[s]
    else:
        sess_clicks[s] = filseq
        sess_times[s] = filtime
print("After filtering", len(sess_clicks))


# Reduce the total number of sessions based on REDUCTION_FACTOR
original_sess_ids = list(sess_clicks.keys())
n_sessions_to_keep = int(len(original_sess_ids) * REDUCTION_FACTOR)

sampled_sess_ids = random.sample(original_sess_ids, n_sessions_to_keep)

sess_clicks = {sid: sess_clicks[sid] for sid in sampled_sess_ids}
sess_times = {sid: sess_times[sid] for sid in sampled_sess_ids}
sess_date = {sid: sess_date[sid] for sid in sampled_sess_ids}

print(f"Total sessions before reduction: {len(original_sess_ids)}")
print(f"Total sessions after {REDUCTION_FACTOR*100}% reduction: {len(sess_clicks)}")


# Split out test set based on dates (uses the reduced sess_date)
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = maxdate - (86400 * 7)

print('Splitting date', splitdate)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

item_dict = {}

# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
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
        if len(outseq) < 2:
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_times += [sess_times[s]]
    return train_ids, train_dates, train_seqs, train_times

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_times = []
    test_dates = []
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

# Data augmentation (Single continuous slice configuration)
def process_seqs(iseqs, idates, itimes):
    out_seqs = []
    out_dates = []
    out_times = []
    labs = []
    ids = []
    for id, seq, date, tim in zip(range(len(iseqs)), iseqs, idates, itimes):
        labs.append(seq[-1])
        out_seqs.append(seq[:-1])
        out_dates.append(date)
        out_times.append(tim[:-1])
        ids.append(id)
    return out_seqs, out_dates, out_times, labs, ids

tr_seqs, tr_dates, tr_times, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates, tra_times)
te_seqs, te_dates, te_times, te_labs, te_ids = process_seqs(tes_seqs, tes_dates, tes_times)

tra = (tr_seqs, tr_labs, tr_times)
tes = (te_seqs, te_labs, te_times)

all_len = 0
for seq in tra_seqs:
    all_len += len(seq)
for seq in tes_seqs:
    all_len += len(seq)
print('max_sess_len', max([len(s) for s in tra_seqs]))
print('avg length: ', all_len/(len(tra_seqs) + len(tes_seqs) * 1.0))

# --- FINAL STEP: Save to the new, reduced directory ---
if not os.path.exists(NEW_DATASET_FOLDER):
    os.makedirs(NEW_DATASET_FOLDER)

def save_as_human_readable(data_list, filename):
    with open(filename, 'w') as f:
        for sequence in data_list:
            if not isinstance(sequence, (list, tuple)) or isinstance(sequence, str):
                sequence = [sequence]
            line = ' '.join(map(str, sequence))
            f.write(line + '\n')
    print(f"Guardado exitosamente como texto legible en: {filename}")

save_as_human_readable(tra, f'{NEW_DATASET_FOLDER}/train_txt.txt')
save_as_human_readable(tes, f'{NEW_DATASET_FOLDER}/test_txt.txt')
# CORREGIDO: Se reemplaza la variable fantasma 'seq' por 'tra_seqs'
save_as_human_readable(tra_seqs, f'{NEW_DATASET_FOLDER}/all_train_seq_txt.txt')

pickle.dump(tra, open(f'{NEW_DATASET_FOLDER}/train.txt', 'wb'))
pickle.dump(tes, open(f'{NEW_DATASET_FOLDER}/test.txt', 'wb'))
pickle.dump(tra_seqs, open(f'{NEW_DATASET_FOLDER}/all_train_seq.txt', 'wb'))

print(f'\nDone. New reduced dataset saved in the "{NEW_DATASET_FOLDER}" folder.')