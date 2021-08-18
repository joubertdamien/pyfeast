import pyfeast as f
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import cv2

def displayNeur(n, w, neur_state, k):
    """
        n: number of neurons
        w: half ROI
        neur_state: neurons variables
        k: Figure index
    """
    size = (2 * w + 1)*(2 * w + 1)
    h = np.sqrt(n).astype(np.int32) + 1
    plt.figure(k)
    plt.clf()
    for i in range(0, n, 1):
        weig = neur_state["weights"][size*i:size*(i+1)].reshape((2 * w + 1), (2 * w + 1))
        plt.subplot(h, h, i + 1)
        plt.imshow(weig)
        plt.axis('off')
        plt.colorbar()
        plt.title(i)
    plt.draw()
    plt.pause(0.001)

def displayThs(n, neur_state, k):
    entropy = -np.sum(np.log(neur_state["ths"]) * neur_state["ths"])
    plt.figure(k)
    plt.clf()
    plt.bar(np.arange(0, n), neur_state["ths"])
    plt.xlabel("Neron ID")
    plt.ylabel("th")
    plt.title('Entropy: {}'.format(entropy))
    plt.ylim([0, 1])
    plt.draw()
    plt.pause(0.001)

def displayAct(n, neur_ev, x, y, k):
    plt.figure(k)
    plt.clf()
    plt.subplot(2, 1, 1)
    for i in range(0, n, 1):
        ind = np.where(neur_ev["n"] == i)
        plt.plot(neur_ev["t"][ind], np.full(len(ind[0]), i), '+')
    plt.xlabel("time")
    plt.ylabel("neuron ID")
    plt.ylim([-1, n + 1])
    plt.subplot(2, 1, 2)
    im = np.zeros((x, y, 3))
    im[neur_ev["y"], neur_ev['x'], 0] = (neur_ev["n"] + 1) * 200 / (n+1) 
    im[neur_ev["y"], neur_ev['x'], 1:] = 255
    for i in range(n):
        im[:10, i*10:(i+1)*10, 0] = (i + 1) * 200 / (n+1) 
        im[:10, i*10:(i+1)*10, 1:] = 255
        #cv2.addText(im, '{}'.format(i), (int(i*10), 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    img_rgb = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_HSV2RGB)
    plt.imshow(img_rgb)
    plt.draw()
    plt.pause(0.001)


def displayEnt(n, neur_state, k):
    plt.figure(k)
    plt.clf()
    plt.bar(np.arange(0, n), neur_state["ent"])
    plt.xlabel("Neron ID")
    plt.ylabel("Entropy")
    plt.draw()
    plt.pause(0.001)


def viewEvent(chunk, tsurface, tw, wait=1):
    tsurface[chunk['y'], chunk['x']] = chunk['t']
    img = 255 * np.exp(-(chunk["t"][-1]- tsurface).astype(np.float32)/(3 * tw))
    img_c = cv2.applyColorMap(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB), cv2.COLORMAP_CIVIDIS)
    cv2.imshow("Time Surface", img_c)
    cv2.waitKey(wait)


def viewEventPola(chunk, tsurface, indexsurface, tw, wait=1):
    tsurface[chunk['y'], chunk['x']] = chunk['t']
    indexsurface[chunk['y'], chunk['x']] = chunk['on'] * 2 -1 
    img = 125 * np.exp(-(chunk["t"][-1]- tsurface).astype(np.float32)/(3 * tw)) * indexsurface + 125
    img_c = cv2.applyColorMap(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB), cv2.COLORMAP_CIVIDIS)
    cv2.imshow("Time Surface", img_c)
    cv2.waitKey(wait)