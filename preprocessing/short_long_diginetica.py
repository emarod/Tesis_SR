import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import random
import pandas as pd
import numpy as np

REDUCTION_FACTOR = 1 # Keep 25% of the data
NEW_DATASET_FOLDER = 'diginetica_weights_pandas_matrix'
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
# --- 0. Definición de Tipos de Datos (Ajustado para Diginetica) ---
optimized_dtypes = {
    # El delimitador del CSV de Diginetica es ';'
    "sessionId": 'int32', 
    "itemId": 'int32',
    # 'timeframe' son milisegundos transcurridos, debe ser un entero
    "timeframe": 'int32', 
    # 'eventdate' es la fecha (YYYY-MM-DD), la dejaremos como string para luego parsear
    "eventdate": 'string' 
}

# --- 1. Lectura, Conversión y Ordenamiento ---
print("Cargando y preprocesando datos de Diginetica...")

data = pd.read_csv(
    filepath_or_buffer=data_path + dataset,
    delimiter=";", # <--- CAMBIO: Delimitador ;
    dtype=optimized_dtypes,
    # Solo leer las columnas necesarias para mayor velocidad (si hubiera más)
    usecols=['sessionId', 'itemId', 'timeframe', 'eventdate'] 
)

# A. Conversión de eventdate a Unix Timestamp (para sess_date)
# Se toma solo la fecha (YYYY-MM-DD), no hay HORA en esta columna.
data['session_date'] = pd.to_datetime(data['eventdate'])
data['unix_date'] = (data['session_date'].values.astype(np.int64) // 10**9)

# B. Liberar la columna 'session_date' y 'eventdate'
del data['session_date'] 
del data['eventdate'] 

# C. ORDENAMIENTO CRÍTICO
# Antes de agrupar, debemos garantizar que las filas de cada sesión
# estén ordenadas por el tiempo transcurrido (timeframe).
# Esto reemplaza el sorted() manual del código original.
data = data.sort_values(by=['sessionId', 'timeframe'])

# --- 2. Agrupación y Conversión a Diccionarios ---
print("Agrupando y convirtiendo a diccionarios...")

# sess_clicks: Lista de itemId por sessionId (ya ordenados por timeframe)
sess_clicks = data.groupby('sessionId')['itemId'].agg(list).to_dict()

# sess_times: Lista de timeframe (milisegundos) por sessionId (ya ordenados)
# Usamos 'timeframe' para esto.
sess_times = data.groupby('sessionId')['timeframe'].agg(list).to_dict()

# sess_date: Última fecha Unix (max) por sessionId
# Esto simula la lógica del código original de tomar la fecha del último evento.
sess_date = data.groupby('sessionId')['unix_date'].max().to_dict()

# Borrar la columna auxiliar 'unix_date'
del data['unix_date']

print("sess_clicks, sess_times y sess_date generados")

# print(sess_times)
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


# Reduce the total number of sessions to 25%

original_sess_ids = list(sess_clicks.keys())
n_sessions_to_keep = int(len(original_sess_ids) * REDUCTION_FACTOR)

# Randomly sample the session IDs to keep
sampled_sess_ids = random.sample(original_sess_ids, n_sessions_to_keep)

# Create the reduced session dictionaries
reduced_sess_clicks = {sid: sess_clicks[sid] for sid in sampled_sess_ids}
reduced_sess_time = {sid: sess_times[sid] for sid in sampled_sess_ids}
reduced_sess_date = {sid: sess_date[sid] for sid in sampled_sess_ids}

# Replace the full dictionaries with the reduced ones
sess_clicks = reduced_sess_clicks
sess_times = reduced_sess_time
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
print(len(sess_times))
# print(sess_date)

##################################################### HERE #####################################

# # Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}

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
        train_times += [sess_times[s]]
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
        test_times += [sess_times[s]]
    return test_ids, test_dates, test_seqs, test_times

tra_ids, tra_dates, tra_seqs, tra_times = obtian_tra()
tes_ids, tes_dates, tes_seqs, tes_times = obtian_tes()


# data augmentation
def process_seqs(iseqs, idates, itimes, avg_len=70):
    short_item_set = set()
    long_item_set = set()
    s_out_seqs, s_out_dates, s_out_times, s_labs, s_ids = [], [], [], [], []
    l_out_seqs, l_out_dates, l_out_times, l_labs, l_ids = [], [], [], [], []
    
    for id, seq, date, tim in zip(range(len(iseqs)), iseqs, idates, itimes):
        if len(seq) - 1 <= avg_len:
            short_item_set.update(seq)
        else:
            short_item_set.update(seq[:avg_len+1])
            long_item_set.update(seq)

        for i in range(1, len(seq)):
            sub_seq = seq[:-i]
            if len(sub_seq) <= avg_len:
                tar = seq[-i]
                s_labs += [tar]
                s_out_seqs += [sub_seq]
                s_out_dates += [date]
                s_out_times += [tim[:-i]]
                s_ids += [id]

                # short_item_set.add(tar)
                # short_item_set.update(sub_seq)
            else:
                tar = seq[-i]
                l_labs += [tar]
                l_out_seqs += [sub_seq]
                l_out_dates += [date]
                l_out_times += [tim[:-i]]
                l_ids += [id]

                # long_item_set.add(tar)
                # long_item_set.update(sub_seq)
    return (s_out_seqs, s_out_dates, s_out_times, s_labs, s_ids, short_item_set), (l_out_seqs, l_out_dates, l_out_times, l_labs, l_ids, long_item_set)

def generate_final_sessions(tra_info, tes_info, folder):
    tr_seqs, tr_dates, tr_times, tr_labs, tr_ids, tr_items_unique = tra_info
    te_seqs, te_dates, te_times, te_labs, te_ids, te_items_unique = tes_info

    all_unique_items = tr_items_unique.union(te_items_unique)

    print(f"Total unique nodes: {len(all_unique_items)}")

    print("training sequences: ", len(tr_seqs))
    print("test sequences: ", len(te_seqs))
    
    tra = (tr_seqs, tr_labs, tr_times)
    tes = (te_seqs, te_labs, te_times)
    
    print('train_test')
    print(len(tr_seqs))
    print(len(te_seqs))
    print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
    print(te_seqs[:3], te_dates[:3], te_labs[:3])
    all = 0
    
    for seq in tra_seqs:
        all += len(seq)
    for seq in tes_seqs:
        all += len(seq)
    print('max_sess_len', max(max([len(s) for s in tr_seqs]), max([len(s) for s in te_seqs])))
    print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
    print('all clicks:', all)
    
    # --- FINAL STEP: Save to the new, reduced directory ---
    if not os.path.exists(folder):
        os.makedirs(folder)
    
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
    
    
    save_as_human_readable(tra, f'{folder}/train_txt.txt')
    save_as_human_readable(tes, f'{folder}/test_txt.txt')
    save_as_human_readable(seq, f'{folder}/all_train_seq_txt.txt')
    
        
    pickle.dump(tra, open(f'{folder}/train.txt', 'wb'))
    pickle.dump(tes, open(f'{folder}/test.txt', 'wb'))
    pickle.dump(tra_seqs, open(f'{folder}/all_train_seq.txt', 'wb'))
    
    print(f'\nDone. New reduced dataset saved in the "{folder}" folder.')
    
    # if not os.path.exists('Nowplaying'):
    #     os.makedirs('Nowplaying')
    # pickle.dump(tra, open('Nowplaying/train.txt', 'wb'))
    # pickle.dump(tes, open('Nowplaying/test.txt', 'wb'))
    # pickle.dump(tra_seqs, open('Nowplaying/all_train_seq.txt', 'wb'))

short_info_tra, long_info_tra = process_seqs(tra_seqs, tra_dates, tra_times)
short_info_tes, long_info_tes = process_seqs(tes_seqs, tes_dates, tes_times)
generate_final_sessions(short_info_tra, short_info_tes, "diginetica_short")
generate_final_sessions(long_info_tra, long_info_tes, "diginetica_long")
print(len(item_dict.keys()))