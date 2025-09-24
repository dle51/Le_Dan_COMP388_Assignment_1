import matplotlib.pyplot as plt

import utils.utils as utils
from sklearn import metrics
import time


def main():
    # Loading data
    f1 = f2 = f3 = None
    try:
        f1 = utils.load_data("src/data/s1.csv")
        f2 = utils.load_data("src/data/s2.csv")
        f3 = utils.load_data("src/data/s3.csv")
    except FileNotFoundError:
        print("Data files not found.")
    assert f1 is not f2 is not f3 is not None
    
    # Question 2.1
    print("Question 2.1")
    f1_fnmr, f1_fmr, f1_eer = utils.compute_sim_fmr_fnmr_eer(f1)
    f2_fnmr, f2_fmr, f2_eer = utils.compute_sim_fmr_fnmr_eer(f2)
    f3_fnmr, f3_fmr, f3_eer = utils.compute_sim_fmr_fnmr_eer(f3)

    print(f"Data Set 1 - FNMR: {f1_fnmr:.3f} FMR: {f1_fmr:.3f} EER: {f1_eer:.3f}")
    print(f"Data Set 2 - FNMR: {f2_fnmr:.3f} FMR: {f2_fmr:.3f} EER: {f2_eer:.3f}")
    print(f"Data Set 3 - FNMR: {f3_fnmr:.3f} FMR: {f3_fmr:.3f} EER: {f3_eer:.3f}")

    # Question 2.3
    f1_d_prime = utils.compute_d_prime(f1)
    f2_d_prime = utils.compute_d_prime(f2)
    f3_d_prime = utils.compute_d_prime(f3)

    print(f"Data Set 1 d\': {f1_d_prime:.3f}")
    print(f"Data Set 2 d\': {f2_d_prime:.3f}")
    print(f"Data Set 3 d\': {f3_d_prime:.3f}")

    print("Question 2.1 Completed.")

    # Question 2.6
    print("Question 2.6")

    f1_x = []
    f1_y = []
    f2_x = []
    f2_y = []
    f3_x = []
    f3_y = []

    for tup in f1:
        f1_x.append(tup[0])
        f1_y.append(tup[1])
    for tup in f2:
        f2_x.append(tup[0])
        f2_y.append(tup[1])
    for tup in f3:
        f3_x.append(tup[0])
        f3_y.append(tup[1])

    arrays = [[f1, f1_x, f1_y], [f2, f2_x, f2_y], [f3, f3_x, f3_y]]
    deltas = []
    for array in arrays:
        naive_start = time.time()
        utils.compute_sim_fmr_tmr_auc(array[0])
        naive_end = time.time()
        naive_delta = (naive_end - naive_start)

        sklearn_start = time.time()
        fpr, tpr, thresholds = metrics.roc_curve(array[1], array[2], pos_label=1)
        metrics.auc(fpr, tpr)
        sklearn_end = time.time()
        sklearn_delta = (sklearn_end - sklearn_start)

        deltas.append([naive_delta, sklearn_delta])

    naive_total_runtime = deltas[0][0] + deltas[1][0] + deltas[2][0]
    sklearn_total_runtime = deltas[0][1] + deltas[1][1] + deltas[2][1]

    print(f"Deltas: {deltas}\nNaive Time: {naive_total_runtime:.3f}\nSKLearn Time: {sklearn_total_runtime:.3f}")

    print("Question 2.6 Completed.")

if __name__ == "__main__":
    main()
