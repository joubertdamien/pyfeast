import event_stream
import pyfeast as f
import numpy as np
import sys
import time
from tqdm import tqdm

file = "data/ev.es"
out_type = np.array([1, 1, 1, 1], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('n', 'u2')])
f.init(720, 1280, 5, 100000, 0.003, 0.001, 0.00001, 10, "f")
n = 100
times = np.zeros(n)
for i in tqdm(range(0, n, 1)):
    decoder = event_stream.Decoder(file)
    start = time.time()
    nb_events = 0
    for chunk in decoder:
        nb_events += chunk.shape[0]
        res = f.filter(chunk, out_type, "f")
    times[i] = time.time() - start
print('Processing rate: {}  ev/s = {} / {} +/- {}'.format(nb_events / np.mean(times), nb_events, np.mean(times), np.std(times)))