rule all:
    input:
        r'notebooks\01_exploratory_data_analysis_dummy.py',
        r'notebooks\02_compare_models_dummy.py',
        r'results\submission.csv'

rule download:
    output:
        r'data\raw\gender_submission.csv',
        r'data\raw\test.csv',
        r'data\raw\train.csv'
    shell:
        r'titanic download'

rule eda:
    input:
        r'data\raw\train.csv',
        r'data\raw\test.csv',
        r'references\data_dict.xlsx',
        r'notebooks\01_exploratory_data_analysis_dummy.ipynb'
    output:
        r'notebooks\01_exploratory_data_analysis_dummy.py'
    shell:
        r'titanic eda'

rule features:
    input:
        r'data\raw\test.csv',
        r'data\raw\train.csv'
    output:
        r'data\processed\X_train.pickle',
        r'data\processed\X_test.pickle',
        r'data\processed\y_train.pickle'
    shell:
        r'titanic features'

rule crossval:
    input:
        r'data\processed\X_train.pickle',
        r'data\processed\y_train.pickle'
    output:
        r'models\logreg.pickle',
        r'models\forest.pickle',
        r'models\svc.pickle'
        r'models\voting.pickle'
    shell:
        r'titanic crossval'

rule compmod:
    input:
        r'models\logreg.pickle',
        r'models\forest.pickle',
        r'models\svc.pickle',
        r'models\voting.pickle'
        r'notebooks\02_compare_models_dummy.ipynb'
    output:
        r'notebooks\02_compare_models_dummy.py'
    shell:
        r'titanic compmod'

rule submission:
    input:
        r'models\forest.pickle'
    output:
        r'results\submission.csv'
    shell:
        r'titanic submission'

rule clean:
    shell:
        r'titanic clean -a'
