from utils import *
import os
import numpy as np
import event_stream

path = "/home/joubertd/Documents/Data/Simulations/SpinningNoise"
noises = [0.05] #[0, 0.01, 0.03, 0.07, 0.09, 0.11, 0.2, 0.5]  #[0, 0.05, 0.5, 5]
shape = [260, 346]
nb_neur = 20
w = 2
neur = ['f']  # ['f', 'c', '1', '2']
out_type = np.array([1, 1, 1, 1], dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('n', 'u2')])
snr = np.zeros(len(noises))
m_entropy = np.zeros((len(noises), len(neur)))
for j, nois in enumerate(noises):
    decoder = event_stream.IndexedDecoder(os.path.join(path, 'Noise_{}Hz.es'.format(nois)), 200000)
    keyframes = decoder.keyframes()
    for n in neur:
        f.init(shape[0], shape[1], w, 50000, 0.001, 0.001, 0.01, nb_neur, n)
    start = time.time() 
    nb_events = 0
    tsurface = np.zeros((shape[0], shape[1]), dtype=np.uint64)
    for keyframe_index in range(0, keyframes):
        chunk = decoder.chunk(keyframe_index)
        nb_events += chunk.shape[0]
        for i, n in enumerate(neur):
            res = f.filter(chunk, out_type, n)
            viewEvent(chunk, tsurface, 20000, 1)
            neur_state = f.getNeuronState(n)
            displayNeur(nb_neur, w, neur_state, 2*i)
            displayThs(nb_neur, neur_state, 2*i+1)
    snr[j] = nb_events
    for i, n in enumerate(neur):
        neur_state = f.getNeuronState(n)
        m_entropy[j, i] = np.mean(neur_state['ent'])
        displayNeur(nb_neur, w, neur_state, 3*i)
        displayThs(nb_neur, neur_state, 3*i+1)
        displayEnt(nb_neur, neur_state, 3*i+2)
    viewEvent(chunk, tsurface, 20000, -1)
print(snr)
snr /= snr[0]
snr = 10 * np.log(snr[0] / (snr - snr[0])) / np.log(10)
print(snr) 
plt.figure()
for j, n in enumerate(neur):
    plt.plot(snr[::-1], m_entropy[::-1, j], "p-", label=n)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Entropy')
    plt.legend()
plt.show()
