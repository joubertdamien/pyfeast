import event_stream
import pyfeast as f
import numpy as np
import sys
import time

file = "data/ev.es"
out_type = np.array([1, 1, 1, 1], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('n', 'u2')])
decoder = event_stream.Decoder(file)
f.init(1280, 720, 5, 100000, 0.003, 0.001, 0.0001, 10)
start = time.time()
nb_events = 0
for chunk in decoder:
    nb_events += chunk.shape[0]
    res = f.filter(chunk, out_type)
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_events / dur, nb_events, dur))
