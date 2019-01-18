import pytest
import os
import pickle
import nbformat as nbf
from titanic.tools import serialize, deserialize, run_ipynb, clean_directory


def test_serialize(tmpdir):
    obj = [1, 2, 3]
    file_path = tmpdir.join('file.pickle')
    serialize(obj, file_path)
    with open(file_path, 'rb') as f:
        obj_deserialized = pickle.load(f)
    assert obj == obj_deserialized


def test_deserialize(tmpdir):
    obj = [1, 2, 3]
    file_path = tmpdir.join('file.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    deserialize(file_path)
    obj_deserialized = deserialize(file_path)
    assert obj == obj_deserialized


def test_run_ipynb(tmpdir):
    nb = nbf.v4.new_notebook()
    text = '# Automatic Jupyter Notebook\nThis is an auto-generated notebook.'
    code = '%pylab inline\nhist(normal(size=2000), bins=50);'
    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                   nbf.v4.new_code_cell(code)]
    file_path = tmpdir.join('notebook.ipynb')
    nbf.write(nb, str(file_path))
    print(file_path)
    run_ipynb(str(file_path))
    file_path_old = tmpdir.join('.old').join('notebook.ipynb.old')
    assert tmpdir.join('.old').isdir()
    assert file_path_old.isfile()
    assert os.path.getsize(file_path_old) > 100
    assert file_path.isfile()
    assert os.path.getsize(file_path) > 1000


@pytest.fixture
def mock_dir(tmpdir):
    tmpdir.mkdir('models').mkdir('forest').join('1.log').write(11111)
    tmpdir.join('models').mkdir('logreg').join('2.log').write(22222)
    tmpdir.join('.gitkeep').write('')
    return tmpdir


def test_clean_directory(mock_dir):
    clean_directory(mock_dir, keep_nested_dirs=True, files_to_keep=['.gitkeep'])
    assert mock_dir.join('models/forest').isdir()
    assert len(os.listdir(mock_dir.join('models/logreg'))) == 0
    assert mock_dir.join('.gitkeep').isfile()
