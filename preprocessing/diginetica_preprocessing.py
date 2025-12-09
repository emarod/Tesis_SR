import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import random

REDUCTION_FACTOR = 1 # Keep 25% of the data
NEW_DATASET_FOLDER = 'diginetica'
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
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date

print("-- Reading data @ %ss" % datetime.datetime.now())
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

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
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq
print("After filtering", len(sess_clicks))


# Reduce the total number of sessions to 25%

original_sess_ids = list(sess_clicks.keys())
n_sessions_to_keep = int(len(original_sess_ids) * REDUCTION_FACTOR)

# Randomly sample the session IDs to keep
sampled_sess_ids = random.sample(original_sess_ids, n_sessions_to_keep)

# Create the reduced session dictionaries
reduced_sess_clicks = {sid: sess_clicks[sid] for sid in sampled_sess_ids}
reduced_sess_date = {sid: sess_date[sid] for sid in sampled_sess_ids}

# Replace the full dictionaries with the reduced ones
sess_clicks = reduced_sess_clicks
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

# # # Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())


# # Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# # Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    # ... (Your original obtian_tra logic remains, but runs on the reduced tra_sess) ...
    train_ids = []
    train_seqs = []
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
    return train_ids, train_dates, train_seqs

# # # Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    # ... (Your original obtian_tes logic remains, but runs on the reduced tes_sess 
    # and uses the smaller item_dict) ...
    test_ids = []
    test_seqs = []
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
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

# Data augmentation
def process_seqs(iseqs, idates):
    # ... (Your original process_seqs logic remains) ...
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
print("#####TR IDS: ", len(set(tr_ids)))
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

all_len = 0
for seq in tra_seqs:
    all_len += len(seq)
for seq in tes_seqs:
    all_len += len(seq)
print('avg length: ', all_len/(len(tra_seqs) + len(tes_seqs) * 1.0))

# --- FINAL STEP: Save to the new, reduced directory ---
if not os.path.exists(NEW_DATASET_FOLDER):
    os.makedirs(NEW_DATASET_FOLDER)
    
pickle.dump(tra, open(f'{NEW_DATASET_FOLDER}/train.txt', 'wb'))
pickle.dump(tes, open(f'{NEW_DATASET_FOLDER}/test.txt', 'wb'))
pickle.dump(tra_seqs, open(f'{NEW_DATASET_FOLDER}/all_train_seq.txt', 'wb'))

print(f'\nDone. New reduced dataset saved in the "{NEW_DATASET_FOLDER}" folder.')