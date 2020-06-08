import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | NP VP Conj VP
IC -> N V | NP VP | N VP | NP V | NP VP AdvP | NP VP NP | N VP NP | N V NP PP
DC -> PP V PP | VP PP | VP | PP PP | PP V PP | P NP
NP -> N | Det NP | AP NP | NP PP
VP -> V | V PP | V NP | V NP PP | VP Adv | Adv VP
AP -> Adj | Adj AP
AdvP -> Adv Adv
PP -> P NP | P S
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    token = nltk.word_tokenize(sentence)

    remember = []

    for i in range(len(token)):
        token[i] = token[i].lower()

        for letter in token[i]:
            if letter.isalpha():
                break
            else:
                remember.append(i)

    if len(remember) != 0:
        for r in remember:
            del token[r]

    return token


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrase = []

    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            if not subNP(subtree):
                noun_phrase.append(subtree)

    return noun_phrase

def subNP(subtree):
    for branch in next(subtree.subtrees()):
        if branch.label() == "NP":
            return True
    return False

if __name__ == "__main__":
    main()
