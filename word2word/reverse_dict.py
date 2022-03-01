import argparse

parser = argparse.ArgumentParser(description="Create an inverse dictionary from an existing file.")
parser.add_argument('filename', type=str, required=True,
                    help="path to the dictionary file containing word translations (1 pair per line)")
args = parser.parse_args()

f = open(args.filename)
rev_mapping = {}

for line in f:
    s, t = line.split()
    try:
        rev_mapping[t].append(s)
    except KeyError:
        rev_mapping[t] = [s]

f.close()

g = open(args.filename + "_inv")

for w, trans in rev_mapping.items():
    for t in trans[:5]:
        g.write(f"{w} {t}\n")

g.close()

