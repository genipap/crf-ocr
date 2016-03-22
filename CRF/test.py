# Q2.b
from lib import *
import numpy as np
import time

if __name__ == "__main__":
    w = np.loadtxt("result/learnt_w.txt", dtype=np.float64)
    t = np.loadtxt("result/learnt_t.txt", dtype=np.float64)
    data = load_data("data/test.txt")
    predictions = []
    correct_letter, correct_word = 0, 0
    start_time = time.time()
    for x, y in data:
        infer = max_sum(x, w, t).tolist()
        correct_letter += sum(1 if infer[i] == y[i] else 0 for i in range(0, len(y)))
        if infer == y.tolist():
            correct_word += 1
        predictions += infer
    print("testing time={}".format(time.time() - start_time))
    print("letter-wise accuracy={}%".format(100 * correct_letter / len(predictions)))
    print("word-wise accuracy={}%".format(100 * correct_word / len(data)))
    with open("result/prediction.txt", mode="w") as f:
        for i in predictions:
            f.write(str(i) + "\n")

# C=1000
# letter-wise accuracy=83.75066798992289%
# word-wise accuracy=47.223029950567025%
