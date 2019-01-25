"""Delete all automatically generated files within the project directory."""

from titanic.tools import clean_directory


def main(allfiles=False, data=False, logs=False, models=False, results=False):
    if allfiles:
        data, logs, models, results = True, True, True, True
    if data:
        clean_directory(r'data', files_to_keep=[r'.gitkeep'])
    if logs:
        pass
        clean_directory(r'logs', files_to_keep=[r'.gitkeep', r'old_logs.zip'])
    if models:
        clean_directory(r'models', files_to_keep=[r'.gitkeep'])
    if results:
        clean_directory(r'results', files_to_keep=[r'.gitkeep'])


if __name__ == '__main__':
    main(allfiles=True)
