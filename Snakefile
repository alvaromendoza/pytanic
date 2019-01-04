from pathlib import Path

rule all:
    input:
        r"data\raw\gender_submission.csv",
        r"data\raw\test.csv",
        r"data\raw\train.csv",

rule download_data:
    output:
        r"data\raw\gender_submission.csv",
        r"data\raw\test.csv",
        r"data\raw\train.csv"
    shell:
        "cd src\titanic & python download_data.py"

rule run_eda:
    input:
        r"data\raw\train.csv",
        r"notebooks\pups.ipynb"
    output:
        "notebooks\.snakemake\pups.tkn"
    shell:
        ''.join([
                "mkdir notebooks\.snakemake & ",
                "echo > notebooks\.snakemake\pups.tkn &",
                "cd src/titanic & python run_eda.py"
                ]) 
#        r"mklink /H dummylink {input[1]}"
#        r"mklink /H dummylink {input[1]} & cd src/titanic & python run_eda.py"
#        r"cd src/titanic & python run_eda.py"
