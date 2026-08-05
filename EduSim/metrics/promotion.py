
import numpy as np
from longling import as_list as as_list


def as_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.asarray(as_list(obj))


def promotion_report(initial_score, final_score, full_score=None, path_length=None, average=True, metrics=None,
                     weights=None):
    metrics = {"absp", "absp_rate", "relp", "relp_rate", "norm_relp", "norm_relp_rate"} if metrics is None else metrics
    ret = {}

    initial_score = as_array(initial_score)
    final_score = as_array(final_score)

    absp = final_score - initial_score

    if weights is not None:
        absp *= as_array(weights)

    if "absp" in metrics:
        ret["absp"] = absp
    if path_length is not None and "absp_rate" in metrics:
        absp_rate = absp / as_array(path_length)
        absp_rate[absp_rate == np.inf] = 0
        ret["absp_rate"] = absp_rate

    if full_score is not None:
        full_score = as_array(full_score)

        if "relp" in metrics:
            relp = absp / full_score
            ret["relp"] = relp

        if path_length is not None and "relp_rate" in metrics:
            relp_rate = absp / (full_score * path_length)
            relp_rate[relp_rate == np.inf] = 0
            ret["relp_rate"] = relp_rate

        if "norm_relp" in metrics:
            ret["norm_relp"] = absp / (full_score - initial_score)
        if path_length is not None and "norm_relp_rate" in metrics:
            norm_relp_rate = absp / ((full_score - initial_score) * path_length)
            norm_relp_rate[norm_relp_rate == np.inf] = 0
            ret["norm_relp_rate"] = norm_relp_rate

    if average:
        return {k: np.average(v) for k, v in ret.items()}
    else:
        return {k: v.tolist() for k, v in ret.items()}
