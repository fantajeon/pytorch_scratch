
import time
import numpy as np
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

def test(q):
    while True:
        data = q.get()
        print("recv_q", data)


if __name__ == "__main__":
    q = mp.Queue(maxsize=1)
    p = mp.Process(target=test, args=(q,))
    p.start()

    for _ in range(10):
        data = np.random.rand(1,10)
        time.sleep(0.1)
        q.put(data)
    p.join()
