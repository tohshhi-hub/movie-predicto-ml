from src.preprocess import load_and_preprocess

def test_preprocessing():
    X, y = load_and_preprocess("data/movies.csv")
    assert X.shape[0] == len(y)
    assert X.shape[1] > 0
