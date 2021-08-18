from utils import *
import os
import numpy as np
import event_stream

file = "data/spinning2.es"
shape = [260, 346]
nb_neur = 10
w = 2
n = 'f'
out_type = np.array([1, 1, 1, 1], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('n', 'u2')])
decoder = event_stream.IndexedDecoder(file, 2000000)
keyframes = decoder.keyframes()
f.init(shape[0], shape[1], w, 50000, 0.01, 0.001, 0.01, nb_neur, n)
start = time.time() 
nb_events = 0
tsurface = np.zeros((shape[0], shape[1]), dtype=np.uint64)
indexsurface = np.zeros((shape[0], shape[1]), dtype=np.int8)
for keyframe_index in range(0, keyframes):
    chunk = decoder.chunk(keyframe_index)
    nb_events += chunk.shape[0]
    res = f.filter(chunk, out_type, n)
    neur_state = f.getNeuronState(n)
viewEventPola(chunk, tsurface, indexsurface, 100000)
displayNeur(nb_neur, w, neur_state, 1)
displayThs(nb_neur, neur_state, 2)
displayAct(nb_neur, res["ev"], shape[0], shape[1], 3)
cv2.waitKey()
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_events / dur, nb_events, dur))
