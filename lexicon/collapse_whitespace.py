import re

INPUT_FILE = "css10-g2p-dict.txt"

def collapse_whitespace():
    with open(INPUT_FILE, "r") as f:
        for line in f:
            print(re.sub(" +", " ", line))

if __name__ == "__main__":
    collapse_whitespace()
