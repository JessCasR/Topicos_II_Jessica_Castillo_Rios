from pathlib import Path

def train_data_path() -> Path:
    """
    Returns the location of train data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the train data directory
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_file = folder / "data/training.csv"
        if data_file.exists() and data_file.is_file():
            print("Train data directory found in ", data_file)
            return data_file
        else:
            raise Exception("Data not found")
        
def test_data_path() -> Path:
    """
    Returns the location of test data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the test data directory
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_file = folder / "data/test.csv"
        if data_file.exists() and data_file.is_file():
            print("Test data directory found in ", data_file)
            return data_file
        else:
            raise Exception("Data not found")