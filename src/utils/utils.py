"""Auxiliary and Utility Functions"""

import matplotlib.pyplot as plt

"""Sums all the values in the given list,
using pairwise summation to reduce round-off error."""


def _pairwise_sum(values):
    sum = float("NaN")  # nothing computed, returns not-a-number

    if len(values) == 0:
        sum = 0.0  # nothing to sum, returns zero

    elif len(values) == 1:
        sum = float(values[0])  # one element, returns it

    elif len(values) == 2:
        sum = float(values[0] + values[1])  # two elements, returns their sum

    else:
        i = int(len(values) / 2)
        sum = _pairwise_sum(values[0:i]) + _pairwise_sum(
            values[i : len(values)]
        )  # recursive call

    return sum


# Computes the variance of the given values.
def _compute_var(values):
    var = float("NaN")  # nothing computed, returns not-a-number

    if len(values) > 0:
        # mean of values
        mean = _pairwise_sum(values) / len(values)

        # deviations
        deviations = [(v - mean) ** 2.0 for v in values]

        # variance
        var = _pairwise_sum(deviations) / len(deviations)

    return var


# Loads data from the CSV file stored in the given file path.
# Expected file line format: <label>,<score>
# Comment lines starting with "#" will be ignored.
# Output: array of (<label>,<score>) elements.
def load_data(file_path):
    # output
    output = []  # empty content

    # reads each line of the file,
    # ignoring empty lines and the ones starting with '#'
    with open(file_path) as f:
        for line in f:
            content = line.strip().split(",")
            if (
                len(content) > 0 and len(content[0]) > 0 and content[0][0] != "#"
            ):  # valid line; other will be ignored
                label = int(content[0])
                score = float(content[1])

                output.append((label, score))

    return output


# Computes d-prime for the given observations.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
# If either the number of impostors or genuine observations is zero,
# it returns 'NaN' as d-prime.
def compute_d_prime(observations):
    # output
    d_prime = float("NaN")  # nothing computed, returns not-a-number

    # separates genuine and impostor scores
    genuine_scores = []
    impostor_scores = []

    # for each given observation
    for obs in observations:
        if obs[0] == 0:  # impostor
            impostor_scores.append(obs[1])
        else:  # genuine
            genuine_scores.append(obs[1])

    # if there are values for both classes (impostor and genuine)
    if len(genuine_scores) > 0 and len(impostor_scores) > 0:
        # computes mean values
        genuine_mean = _pairwise_sum(genuine_scores) / len(genuine_scores)
        impostor_mean = _pairwise_sum(impostor_scores) / len(impostor_scores)

        # computes variances
        genuine_var = _compute_var(genuine_scores)
        impostor_var = _compute_var(impostor_scores)

        # d-prime computation
        d_prime = (
            2.0**0.5
            * abs(genuine_mean - impostor_mean)
            / (genuine_var + impostor_var) ** 0.5
        )

    return d_prime


# Computes FMR from the given similarity observations,
# according to the given threshold.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
# If the number of impostors is zero, it returns 'NaN' as FMR.
def compute_sim_fmr(observations, threshold):
    fmr = float("NaN")  # nothing computed, returns not-a-number

    # counters
    impostor_count = 0
    false_match_count = 0

    # for each observation
    for obs in observations:
        if obs[0] == 0:  # impostor
            impostor_count = impostor_count + 1
            if obs[1] >= threshold:
                false_match_count = false_match_count + 1

    # FMR computation
    if impostor_count > 0:
        fmr = false_match_count / impostor_count

    return fmr


# Computes FNMR from the given similarity observations,
# according to the given threshold.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
# If the number of genuine observations is zero, it returns 'NaN' as FNMR.
def compute_sim_fnmr(observations, threshold):
    fnmr = float("NaN")  # nothing computed, returns not-a-number

    # counters
    genuine_count = 0
    false_non_match_count = 0

    # for each observation
    for obs in observations:
        if obs[0] != 0:  # genuine observation
            genuine_count = genuine_count + 1

            if obs[1] < threshold:
                false_non_match_count = false_non_match_count + 1

    # FNMR computation
    if genuine_count > 0:
        fnmr = false_non_match_count / genuine_count

    return fnmr


# Computes FNMR and FMR at EER from the given similarity observations.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
# Output: FNMR, FMR, EER_THRESHOLD.
# If either the number of impostors or genuine observations is zero,
# it returns 'NaN', 'NaN', 'NaN'.
def compute_sim_fmr_fnmr_eer(observations):
    # computed FNMR and FMR at EER, and EER threshold
    output_fnmr = float("NaN")  # nothing computed, returns not-a-number
    output_fmr = float("NaN")
    output_threshold = float("NaN")

    # holds the difference between FMR and FNMR
    fnmr_fmr_diff = float("inf")  # a very large float

    # sorted list of scores
    scores = sorted([obs[1] for obs in observations])
    if len(scores) > 0:
        # for each score taken as threshold
        for threshold in scores:
            current_fnmr = compute_sim_fnmr(observations, threshold)
            current_fmr = compute_sim_fmr(observations, threshold)

            # cancels computation if any of the FNMR or FMR values are 'NaN' (impossible to compute them)
            if not float("-inf") < current_fnmr < float("inf") or not float(
                "-inf"
            ) < current_fmr < float("inf"):
                break

            # updates the difference between FNMR and FMR, if it is the case
            current_diff = abs(current_fnmr - current_fmr)
            if current_diff <= fnmr_fmr_diff:
                fnmr_fmr_diff = current_diff

                # updates current values
                output_fnmr = current_fnmr
                output_fmr = current_fmr
                output_threshold = threshold

            else:
                # difference will start to increase, nothing to do anymore
                break

    return output_fnmr, output_fmr, output_threshold


# Computes FMR x TMR (a.k.a. 1.0 - FNMR) AUC from the given similarity observations.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
# Output: AUC, array with FMR values, array with TMR values.
# If either the number of impostors or genuine observations is zero, it returns 'NaN', [], [].
def compute_sim_fmr_tmr_auc(observations):
    # output values
    auc = float("NaN")  # nothing computed, returns not-a-number
    fmrs = []
    tmrs = []

    # sorted list of scores
    scores = sorted([obs[1] for obs in observations])
    if len(scores) > 0:
        # for each score taken as a threshold
        for threshold in scores:
            current_fmr = compute_sim_fmr(observations, threshold)
            current_fnmr = compute_sim_fnmr(observations, threshold)

            # cancels computation if any of the FNMR or FMR values are 'NaN' (impossible to compute them)
            if not float("-inf") < current_fmr < float("inf") or not float(
                "-inf"
            ) < current_fnmr < float("inf"):
                break

            # adds the computed values to the proper lists
            fmrs.append(current_fmr)
            tmrs.append(1.0 - current_fnmr)

        # computes the AUC
        if len(fmrs) > 0 and len(tmrs) > 0:
            # # adds the border points on [0.0, 0.0] and [1.0, 1.0] for completeness
            if fmrs[-1] != 0.0 or tmrs[-1] != 0.0:
                fmrs.append(0.0)
                tmrs.append(0.0)

            if fmrs[0] != 1.0 or tmrs[0] != 1.0:
                fmrs.insert(0, 1.0)
                tmrs.insert(0, 1.0)

            auc_parts = []
            for i in range(len(fmrs) - 1):
                auc_parts.append(
                    abs(fmrs[i] - fmrs[i + 1]) * (tmrs[i] + tmrs[i + 1]) / 2.0
                )
            auc = _pairwise_sum(auc_parts)

    return auc, fmrs, tmrs


# Plots the histograms of the scores of the impostors and of the genuine observations together.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
def plot_hist(observations):
    impostors = []
    genuine = []

    for item in observations:
        if item[0] == 0:
            impostors.append(item[1])
        else:
            genuine.append(item[1])

    plt.xlabel("score")
    plt.ylabel("frequency")

    plt.hist(impostors, facecolor="red", alpha=0.5, label="impostor", align="mid")
    plt.hist(genuine, facecolor="blue", alpha=0.5, label="genuine", align="mid")
    plt.legend(loc="lower right")

    d_prime = compute_d_prime(observations)
    if float("-inf") < d_prime < float("inf"):
        plt.title("Score distribution, d'=" + "{:.2f}".format(d_prime))
    else:
        plt.title("Score distribution")

    plt.show()


# Plots the FMR x TMR AUC from the given similarity observations.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or something else (genuine).
def plot_sim_fmr_tmr_auc(observations):
    plt.xlabel("FMR")
    plt.ylabel("TMR")

    auc, fmrs, tmrs = compute_sim_fmr_tmr_auc(observations)
    if float("-inf") < auc < float("inf"):
        plt.plot(fmrs, tmrs, label="AUC: " + "{:.2f}".format(auc))
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.legend(loc="lower right")

    plt.title("ROC curve")
    plt.show()
