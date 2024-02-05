from pathlib import Path
from typing import Dict
from os import listdir
import json_tricks
import numpy as np

def load_result(filename: str) -> Dict:
    results = json_tricks.load(filename)["results"]
    return results

def load_results(results_dir: Path) -> Dict:
    all_results = dict()
    for f in listdir(str(results_dir)):
        if f.endswith(".json"):
            result = load_result(str(Path(results_dir, f)))
            name = f.split("-")[1].split(".")[0].split("_")[0]
            all_results[name] = result
        else:
            print(f"Skipping {f}. Results are expected to be stored as json files")
    
    return all_results

def get_best_methods(all_results: Dict):
    ests = ['BasicLossEstimator_', 'HaversineNearestNeighborEstimator_', "HaversineFitNearestNeighborEstimator_"]
    best_methods = list()
    for est in ests:
        all_ests = list()
        for method, results in all_results.items(): 
            estimate = results[est]["estimate"]
            all_ests.append(estimate)
        best_methods.append(np.argmin(np.array(all_ests)))
    
    return best_methods

def write_table(all_results: Dict, tablefile: str):
    nrows = len(all_results.keys())
    ncols = len(all_results[list(all_results.keys())[0]].keys())
    row_names = list()
    col_names = list()
    for row in all_results.keys():
        if row == "GeographicallyWeightedRegression_":
            row_names.append("GWR")
        elif row == "KRRSpatialRegression_":
            row_names.append("Spatial GP")
        else:
            print(f"Unexpected model {row}")
    for col in all_results[list(all_results.keys())[0]].keys():
        if col == "BasicLossEstimator_":
            col_names.append("Holdout")
        elif col == "HaversineNearestNeighborEstimator_":
            col_names.append("1NN")
        elif col == "HaversineFitNearestNeighborEstimator_":
            col_names.append("SNN")
        else:
            print(f"Unexpected estimator {col}")
    
    strs = list()
    strs.append("\\begin{tabular}{c|ccc} \n")
    strs.append("    & "+ " & ".join(col_names) + " \\\\ \hline \n")
    best_methods = get_best_methods(all_results)
    for i, (k, v) in enumerate(all_results.items()):
        res = list()
        for j, r in enumerate(v.values()):
            best_method = (best_methods[j] == i)
            est = "${:.2f}".format(r["estimate"]) if not best_method else "$\\mathbf{{{:.2f}".format(r["estimate"])
            if r["confidence"] is not None and not best_method:
                conf = " \pm {:.2f}$".format(2 * r["confidence"])
            elif  r["confidence"] is not None:
                conf = " \pm {:.2f}}}$".format(2 * r["confidence"])  
            elif best_method:
                conf = "}$"
            else: 
                conf = "$"
            string = est + conf
            res.append(string)
        res_str = f"{row_names[i]} & "+ " & ".join(res) + " \\\\ \n"
        strs.append(res_str)
    strs.append("\\end{tabular}")

    with open(tablefile, "w") as f:
        f.writelines(strs)




if __name__ == "__main__":
    # Define results path
    results_dir = Path(Path(__file__).parents[1], "results")
    metrotablefile = Path(Path(__file__).parents[4], "figures", "airTemp", "metro-table.tex")
    geotablefile = Path(Path(__file__).parents[4], "figures", "airTemp", "geo-table.tex")
    # load all data sets in directory
    all_results = load_results(results_dir)
    # write table
    write_table(all_results["metro"], metrotablefile)
    write_table(all_results["grid"], geotablefile)
