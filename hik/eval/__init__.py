import pickle
from typing import Dict, List, Tuple
from hik.eval.evaluator import Evaluator, EvaluationActionType  # noqa F401
from multiprocessing import Pool


def save_results(fname: str, result: Dict):
    """ """
    with open(fname, "wb") as f:
        pickle.dump(result, f)


def load_results(fname: str) -> Dict:
    with open(fname, "rb") as f:
        return pickle.load(f)


def thread_load_and_fix_data(entry: Tuple[str, str, Evaluator]):
    path, name, ev = entry
    results = load_results(path)
    return (ev.legacy_fix_pids_to_results(results), name)


def load_and_fix_parallel(results: List[Tuple[str, str]], ev: Evaluator):
    """
    :param results: [
        (path_to_results.pkl, method_name)
    ]
    """
    data = [(path, name, ev) for (path, name) in results]
    with Pool(len(data)) as p:
        return p.map(thread_load_and_fix_data, data)
