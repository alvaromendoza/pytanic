"""Generate image of DAG of the entire pipeline for README."""

import os
from graphviz import Source


def main():
    dag = """
    digraph snakemake_dag {
        graph[bgcolor=white, margin=0];
        node[shape=box, style=rounded, fontname=helvetica, fontsize=10, penwidth=2];
        edge[penwidth=2, color=grey];
            1[label = "Exploratory analysis\n(Jupter)", color = "0.58 0.6 0.85", style="rounded,dashed"];
            2[label = "Compare models\n(Jupyter)", color = "0.33 0.6 0.85", style="rounded,dashed"];
            3[label = "Create submission", color = "0.50 0.6 0.85", style="rounded"];
            4[label = "Download data", color = "0.08 0.6 0.85", style="rounded"];
            5[label = "Cross-validate models", color = "0.25 0.6 0.85", style="rounded"];
            6[label = "Feature engineering", color = "0.17 0.6 0.85", style="rounded"];
            4 -> 1
            5 -> 2
            5 -> 3
            6 -> 5
            4 -> 6
    }
    """

    s = Source(dag, format='svg')
    s.render('misc/images/dag', view=True)


if __name__ == '__main__':
    os.chdir(r'../../../')
    main()












