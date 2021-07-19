import event_stream
import filter as f
import numpy as np
import sys
import time

file = "data/ev.es"
decoder = event_stream.Decoder(file)
f.init(720, 1280, 5, 10000, 10)
start = time.time()
nb_events = 0
for chunk in decoder:
    nb_events += chunk.shape[0]
    res = f.filter(chunk)
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_events / dur, nb_events, dur))
