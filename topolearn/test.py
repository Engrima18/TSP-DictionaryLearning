import os


def test():
    print("Current Working Directory:", os.getcwd())
    print("File Path:", os.path.join(os.getcwd(), "cache\\DataTS.pkl"))
