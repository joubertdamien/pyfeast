import event_stream
import pyfeast as f
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import cv2

def displayNeur(n, w, neur_state):
    size = (2 * w + 1)*(2 * w + 1)
    h = np.sqrt(n).astype(np.int32) + 1
    plt.figure(1)
    plt.clf()
    for i in range(0, n, 1):
        weig = neur_state["weights"][size*i:size*(i+1)].reshape((2 * w + 1), (2 * w + 1))
        plt.subplot(h, h, i + 1)
        plt.imshow(weig)
        plt.axis('off')
        plt.colorbar()
    plt.draw()
    plt.pause(0.001)

def displayThs(n, neur_state):
    plt.figure(2)
    plt.clf()
    plt.bar(np.arange(0, n), neur_state["ths"])
    plt.xlabel("Neron ID")
    plt.ylabel("th")
    plt.ylim([0, 1])
    plt.draw()
    plt.pause(0.001)

def displayAct(n, neur_ev, x, y):
    plt.figure(3)
    plt.clf()
    plt.subplot(2, 1, 1)
    for i in range(0, n, 1):
        ind = np.where(neur_ev["n"] == i)
        plt.plot(neur_ev["t"][ind], np.full(len(ind[0]), i), '+')
    plt.xlabel("time")
    plt.ylabel("neuron ID")
    plt.ylim([-1, n + 1])
    plt.subplot(2, 1, 2)
    for i in range(0, n, 1):
        ind = np.where(neur_ev["n"] == i)
        plt.plot(neur_ev["x"][ind], neur_ev['y'][ind], '+')
    plt.xlim([0, x])
    plt.xlim([0, y])
    plt.draw()
    plt.pause(0.001)


def viewEvent(chunk, tsurface, tw):
    tsurface[chunk['y'], chunk['x']] = chunk['t']
    img = 255 * np.exp(-(chunk["t"][-1]- tsurface).astype(np.float32)/(3 * tw))
    img_c = cv2.applyColorMap(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB), cv2.COLORMAP_CIVIDIS)
    cv2.imshow("Time Surface", img_c)
    cv2.waitKey(1)

file = "data/spinning2.es"
shape = [260, 346]
nb_neur = 50
w = 5
out_type = np.array([1, 1, 1, 1], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('n', 'u2')])
decoder = event_stream.IndexedDecoder(file, 200000)
keyframes = decoder.keyframes()
f.init(shape[0], shape[1], w, 50000, 0.01, 0.001, 0.01, nb_neur)
start = time.time() 
nb_events = 0
tsurface = np.zeros((shape[0], shape[1]), dtype=np.uint64)
for keyframe_index in range(0, keyframes):
    chunk = decoder.chunk(keyframe_index)
    nb_events += chunk.shape[0]
    res = f.filter(chunk, out_type)
    neur_state = f.getNeuronState()
    viewEvent(chunk, tsurface, 100000)
    displayNeur(nb_neur, w, neur_state)
    displayThs(nb_neur, neur_state)
    displayAct(nb_neur, res["ev"], shape[0], shape[1])
dur = time.time() - start
print('Processing rate: {} = {} / {} ev/s'.format(nb_events / dur, nb_events, dur))
