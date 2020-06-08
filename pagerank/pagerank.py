import os
import copy
import random
import numpy as np
import re
import sys


DAMPING = 0.85
SAMPLES = 10000
DIFF = 0.001

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probab_distribution = {}
    links = len(corpus[page])

    if links:            
        for link in corpus:
            probab_distribution[link] = (1 - damping_factor) / len(corpus)

        for link in corpus[page]:
            probab_distribution[link] += damping_factor / links
    else:
        for link in corpus:
            probab_distribution[link] = 1 / len(corpus)
    
    return probab_distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = {}
    for page in corpus:
        distribution[page] = 0
    
    page = random.choice(list(corpus.keys()))

    for i in range(1, n):
        current_distribution = transition_model(corpus, page, damping_factor)
        for page in distribution:
            distribution[page] = ((i-1) * distribution[page] + current_distribution[page]) / i
        
        page = random.choices(list(distribution.keys()), list(distribution.values()), k=1)[0]

    return distribution

def normalize(output):
    summ = sum(output.values())
    return {k:(v/summ) for (k,v) in output.items()}

def is_converged(prev_pr, next_pr):
    return all([(abs(next_pr[k] - v) <= DIFF) for (k,v) in sorted(prev_pr.items())])

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    keys = list(corpus.keys())
    prev_pagerank = dict(zip(keys, np.ones(len(keys), dtype=np.float)/len(keys)))
    reverse_corpus = {k: set() for k in keys}
    random_val = (1-damping_factor)/len(keys)
    for k,v in corpus.items():
        for page in v:
            reverse_corpus[page].add(k)
    while True:
        next_pagerank = {}
        for k,v in prev_pagerank.items():
            tmp = 0.0
            for page in reverse_corpus[k]:
                tmp += prev_pagerank[page]/len(corpus[page])
            next_pagerank[k] = random_val + damping_factor * tmp
        next_pagerank = normalize(next_pagerank)
        if (is_converged(prev_pagerank, next_pagerank)):
            break
        prev_pagerank = next_pagerank
    return prev_pagerank

if __name__ == "__main__":
    main()
