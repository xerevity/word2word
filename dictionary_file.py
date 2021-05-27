import os
import pickle
from time import time
import itertools as it
import argparse
import operator
from tqdm import tqdm
from word2word import Word2word

from word2word.utils import (
    download_or_load, download_os2018, get_savedir
)
from word2word.tokenization import (
    load_tokenizer, get_sents, get_vocab, update_dicts
)
from word2word.methods import (
    get_trans_co, get_trans_pmi
)


def rerank(x2ys, x2cnt, x2xs, width, n_trans, threshold=0.0):
    """Re-rank word translations by computing CPE scores.

    See paper for details about the CPE method."""
    x2ys_cpe = dict()
    for x, ys in tqdm(x2ys.items()):
        cntx = x2cnt[x]
        y_scores = []
        for y, cnty in sorted(ys.items(), key=operator.itemgetter(1), reverse=True)[:width]:
            ts = cnty / float(cntx)  # translation score: initial value
            if x in x2xs:
                for x2, cntx2 in x2xs[x].items():  # Collocates
                    p_x_x2 = cntx2 / float(cntx)
                    p_x2_y2 = 0
                    if x2 in x2ys:
                        p_x2_y2 = x2ys[x2].get(y, 0) / float(x2cnt[x2])
                    ts -= (p_x_x2 * p_x2_y2)
            y_scores.append((y, ts))
        _ys_ = sorted(y_scores, key=lambda y_score: y_score[1], reverse=True)[:n_trans]
        _ys_ = [y for y in _ys_ if y[1] >= threshold]
        _ys_ = [each[0] for each in _ys_]
        x2ys_cpe[x] = _ys_

    return x2ys_cpe


def _rerank_mp(x_and_ys, shared_inputs, threshold):
    """Internal multiprocessing function for `rerank_fast()`."""
    x, ys = x_and_ys
    x2ys, x2cnt, x2xs, width, n_trans = shared_inputs

    sorted_ys = sorted(ys.items(),
                       key=operator.itemgetter(1),
                       reverse=True)[:width]
    if x not in x2xs:
        return x, [y for y, score in sorted_ys[:n_trans]]

    def _correction(y):
        return sum(
            cntx2 * x2ys[x2][y] / float(x2cnt[x2])
            for x2, cntx2 in x2xs[x].items() if x2 in x2ys and y in x2ys[x2]
        )

    y_scores = [(y, cnty - _correction(y)) for y, cnty in sorted_ys]
    y_scores = sorted(y_scores, key=operator.itemgetter(1), reverse=True)
    y_scores = [y for y in y_scores if y[1] >= threshold]
    reranked_ys = [y for y, score in y_scores[:n_trans]]
    return x, reranked_ys


def rerank_mp(x2ys, x2cnt, x2xs, width, n_trans, num_workers, threshold):
    """Re-rank word translations by computing CPE scores.

    Uses multiprocessing to speed up computation (significantly).
    In Python 3.8+, shared_inputs can be implemented directly as shared_memory.

    See paper for details about the CPE method."""
    from multiprocessing import Pool

    shared_inputs = x2ys, x2cnt, x2xs, width, n_trans
    print(f"Entering multiprocessing with {num_workers} workers..."
          f" (#words={len(x2ys)})")

    with Pool(num_workers) as p:
        x2ys_cpe = dict(p.starmap(
            _rerank_mp,
            zip(x2ys.items(), it.repeat(shared_inputs), it.repeat(threshold)),
        ))
    return x2ys_cpe


class MyWord2word(Word2word):
    """The word2word class modified.

    Added --split option for fast tokenization of pre-processed corpora
    Added --threshold option to only keep the word translations above a score
    New method: write_dict()

    Usage:
        from word2word import Word2word

        # Download and load a pre-computed bilingual lexicon
        en2fr = Word2word("en", "fr")
        print(en2fr("apple"))
        # out: ['pomme', 'pommes', 'pommier', 'tartes', 'fleurs']

        # Build a custom bilingual lexicon
        # (requires two aligned files, e.g., my_corpus.en, my_corpus.fr)
        my_en2fr = Word2word.make("en", "fr", "my_corpus")
    """

    def __init__(self, lang1, lang2, word2x=None, y2word=None, x2ys=None, custom_savedir=None):
        super().__init__(lang1, lang2, word2x, y2word, x2ys, custom_savedir)

    @classmethod
    def make(
            cls,
            lang1: str,
            lang2: str,
            datapref: str = None,
            n_lines: int = 100000000,
            cutoff: int = 5000,
            rerank_width: int = 100,
            rerank_impl: str = "multiprocessing",
            cased: bool = True,
            n_translations: int = 10,
            save_cooccurrence: bool = False,
            save_pmi: bool = False,
            savedir: str = None,
            num_workers: int = 1,
            threshold: float = 1.0,
            split: bool = False
    ):
        """Build a bilingual lexicon using a parallel corpus."""

        print("Step 0. Check files")
        lang1, lang2 = sorted([lang1, lang2])
        if datapref:
            lang1_file, lang2_file = [
                f"{datapref}.{lang}" for lang in [lang1, lang2]
            ]
            assert os.path.exists(lang1_file), \
                f"custom parallel corpus file missing at {datapref}.{lang1}"
            assert os.path.exists(lang2_file), \
                f"custom parallel corpus file missing at {datapref}.{lang2}"
        else:
            lang1_file, lang2_file = download_os2018(lang1, lang2)

        print("Step 1. Load tokenizer")
        tokenizer1 = load_tokenizer(lang1)
        tokenizer2 = load_tokenizer(lang2)

        t0 = time()
        print("Step 2. Constructing sentences")
        sents1 = get_sents(
            lang1_file, lang1, tokenizer1, cased, n_lines, num_workers, split
        )
        sents2 = get_sents(
            lang2_file, lang2, tokenizer2, cased, n_lines, num_workers, split
        )
        print(f"Time taken for step 2: {time() - t0:.2f}s")

        assert len(sents1) == len(sents2), (
            f"{lang1} and {lang2} files must have the same number of lines.\n"
            f"({lang1}: {len(sents1)} lines, {lang2}: {len(sents2)} lines)"
        )

        # input savedir if provided, else datapref (custom data location);
        # system default otherwise
        savedir = get_savedir(savedir if savedir else datapref)

        print("Step 3. Compute vocabularies")
        # word <-> index
        word2x, x2word, x2cnt = get_vocab(sents1)
        word2y, y2word, y2cnt = get_vocab(sents2)

        print("Step 4. Update count dictionaries")
        # monolingual and cross-lingual dictionaries
        x2xs, y2ys, x2ys, y2xs = update_dicts(
            sents1, sents2, word2x, word2y, cutoff
        )

        t0 = time()
        print("Step 5. Translation using CPE scores")
        if rerank_impl == "simple":
            x2ys_cpe = rerank(x2ys, x2cnt, x2xs, rerank_width, n_translations, threshold)
            y2xs_cpe = rerank(y2xs, y2cnt, y2ys, rerank_width, n_translations, threshold)
        elif rerank_impl == "multiprocessing":
            x2ys_cpe = rerank_mp(
                x2ys, x2cnt, x2xs, rerank_width, n_translations, num_workers, threshold
            )
            y2xs_cpe = rerank_mp(
                y2xs, y2cnt, y2ys, rerank_width, n_translations, num_workers, threshold
            )
        else:
            raise ValueError("unrecognized --rerank_impl argument. "
                             "Options: simple, multiprocessing")
        print(f"Time taken for step 5: {time() - t0:.2f}s")

        print("Saving...")
        Word2word.save(lang1, lang2, savedir, word2x, word2y, x2word,
                       x2ys_cpe, y2word, y2xs_cpe)

        if save_cooccurrence:
            print("Step 5-1. Translation using co-occurrence counts")
            subdir = os.path.join(savedir, "co")
            os.makedirs(subdir, exist_ok=True)

            x2ys_co = get_trans_co(x2ys, n_translations)
            y2xs_co = get_trans_co(y2xs, n_translations)
            Word2word.save(lang1, lang2, subdir, word2x, word2y, x2word,
                           x2ys_co, y2word, y2xs_co)

        if save_pmi:
            print("Step 5-2. Translation using PMI scores")
            subdir = os.path.join(savedir, "pmi")
            os.makedirs(subdir, exist_ok=True)

            seqlens1 = [len(sent) for sent in sents1]
            seqlens2 = [len(sent) for sent in sents2]
            Nx = sum(seqlens1)
            Ny = sum(seqlens2)
            Nxy = sum([seqlen_x * seqlen_y
                       for seqlen_x, seqlen_y in zip(seqlens1, seqlens2)])

            x2ys_pmi = get_trans_pmi(x2ys, x2cnt, y2cnt, Nxy, Nx, Ny,
                                     rerank_width, n_translations)
            y2xs_pmi = get_trans_pmi(y2xs, y2cnt, x2cnt, Nxy, Ny, Nx,
                                     rerank_width, n_translations)

            Word2word.save(lang1, lang2, subdir, word2x, word2y, x2word,
                           x2ys_pmi, y2word, y2xs_pmi)

        print("Done!")
        return cls(lang1, lang2, word2x, y2word, x2ys_cpe)

    def write_dict(self):
        """Save a bilingual dictionary containing the translations in text format."""

        f = open(args.datapref + f".{args.lang1}-{args.lang2}.dict.txt", 'w')
        g = open(args.datapref + f".{args.lang2}-{args.lang1}.dict.txt", 'w')

        w12 = Word2word.load(self.lang1, self.lang2, savedir=args.savedir)
        w21 = Word2word.load(self.lang2, self.lang1, savedir=args.savedir)

        for w in w12.word2x.keys():
            for y in w12(w):
                f.write(f"{w} {y}\n")

        for w in w21.word2x.keys():
            for y in w21(w):
                g.write(f"{w} {y}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save a bilingual dictionary "
                                                 "suitable for word embedding mappings like vecmap,"
                                                 "containing the translations with the highest scores from word2word")
    parser.add_argument('--lang1', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--lang2', type=str, required=True,
                        help="ISO 639-1 code of language. "
                             "See `http://opus.nlpl.eu/OpenSubtitles2018.php`")
    parser.add_argument('--datapref', type=str, default=None,
                        help="data prefix to a custom parallel corpus. "
                             "builds a bilingual lexicon using OpenSubtitles2018 "
                             "unless this option is provided.")
    parser.add_argument('--n_lines', type=int, default=100000000,
                        help="number of parallel sentences used")
    parser.add_argument('--cutoff', type=int, default=5000,
                        help="number of words that are used in calculating collocates within each language")
    parser.add_argument('--rerank_width', default=100, type=int,
                        help="maximum number of target-side collocates considered for reranking")
    parser.add_argument('--rerank_impl', default="multiprocessing", type=str,
                        help="choice of reranking implementation: simple, multiprocessing (default)")
    parser.add_argument('--cased', dest="cased", action="store_true",
                        help="Keep the case.")
    parser.add_argument('--n_translations', type=int, default=10,
                        help="number of final word2word translations kept")
    parser.add_argument('--save_cooccurrence', dest="save_cooccurrence", action="store_true",
                        help="Save the cooccurrence results")
    parser.add_argument('--save_pmi', dest="save_pmi", action="store_true",
                        help="Save the pmi results")
    parser.add_argument('--savedir', type=str, default=None,
                        help="location to store bilingual lexicons."
                             "make sure to use this input when loading from "
                             "a custom-built lexicon.")
    parser.add_argument('--num_workers', default=16, type=int,
                        help="number of workers used for multiprocessing")
    # parser.add_argument('--threshold', type=float, default=1.0,
    #                    help="translations below this score will be ignored")
    parser.add_argument('--split', dest="split", action="store_true",
                        help="separate words by simple splitting (recommended if the corpus is pre-processed)")

    args = parser.parse_args()

    w2w = Word2word.make(**vars(args))
    w2w.write_dict()

