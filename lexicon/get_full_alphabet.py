import re

INPUT_FILE = "css10-g2p-dict.txt"

def get_alphabet():
    alphabet = set()
    with open(INPUT_FILE, "r") as dict:
        for line in dict:
            line = clean(line)
            word = line.split(' ')[0]
            phonemes = line.split(' ')[1:]
            if "freudige" in word:
                for char in word:
                    print("{} {}".format(char, ord(char)))
            print(line)
            print(word)
            print(phonemes)
            alphabet.update(phonemes)
    print(alphabet)

def clean(s):
    cleaned_text = s.replace('\n', '').replace('\t', ' ')
    cleaned_text = re.sub(" +", " ", cleaned_text)
    return cleaned_text    

if __name__ == "__main__":
    get_alphabet()
