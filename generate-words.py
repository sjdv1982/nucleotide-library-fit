import numpy as np


class_vector = np.load("class_vector.npy")
NCLASSES = len(class_vector)
prop2d = np.load("propensity_2d.npy")
assert prop2d.shape == (NCLASSES, NCLASSES)
prop3d = np.load("propensity_3d.npy")
assert prop3d.shape == (NCLASSES, NCLASSES, NCLASSES)


def gather_from_triple(triple_min, triple_max):
    g_min = triple_min.min(axis=2)
    g_max = triple_max.max(axis=2)
    return g_min, g_max


def gather_from_tuple(tuple_min, tuple_max):
    g_min = tuple_min.min(axis=1)
    g_max = tuple_max.max(axis=1)
    return g_min, g_max


unit_vec = np.ones(NCLASSES)

# Transition probability X=>Y given X
tuple0 = unit_vec[:, None] * class_vector[None, :] * prop2d
tuple0 /= tuple0.sum(axis=1)[:, None]

# Transition probability Y=>X given Y
tuple0_inv = unit_vec[:, None] * class_vector[None, :] * prop2d.T
tuple0_inv /= tuple0_inv.sum(axis=1)[:, None]


def adjust_class_vector(vec, M):
    # Adjust class_vector as an eigen vector of M
    w, V = np.linalg.eig(M.T)
    idx = np.where(np.abs(w - 1) < 1e-8)[0]

    # basis for left eigenspace (columns)
    B = V[:, idx]  # shape (n, k)

    # orthogonal projection of class_vector onto span(B)
    # (works best if M is diagonalizable and not too ill-conditioned)
    vec_prime = np.real_if_close((B @ np.linalg.pinv(B) @ vec.reshape(-1, 1)).ravel())
    vec_prime /= vec_prime.sum()
    return vec_prime


print("Original class vector")
print(class_vector, class_vector.sum())
print("New class vector 1, adjusted for fwd tuple matrix")
class_vector_prime1 = adjust_class_vector(class_vector, tuple0)
assert np.allclose(class_vector_prime1, class_vector_prime1 @ tuple0)
print(class_vector_prime1, class_vector_prime1.sum())
print("New class vector 2, adjusted for rev tuple matrix")
class_vector_prime2 = adjust_class_vector(class_vector, tuple0_inv)
assert np.allclose(class_vector_prime2, class_vector_prime2 @ tuple0_inv)
print(class_vector_prime2, class_vector_prime2.sum())

print("New class vector, mean of the two")
class_vector_prime = (class_vector_prime1 + class_vector_prime2) / 2
print(class_vector_prime, class_vector_prime.sum())
print("New class vector, multiplied with tuple matrix")
print(class_vector_prime @ tuple0, (class_vector_prime @ tuple0).sum())
print(np.abs(class_vector_prime @ tuple0 - class_vector_prime).max())
print("New class vector, multiplied with rev tuple matrix")
print(class_vector_prime @ tuple0_inv, (class_vector_prime @ tuple0_inv).sum())
print(np.abs(class_vector_prime @ tuple0_inv - class_vector_prime).max())
print()

class_matrix1 = class_vector_prime[:, None] * tuple0
assert np.allclose(class_matrix1.sum(axis=1), class_vector_prime)
class_matrix2 = class_vector_prime[:, None] * tuple0_inv
assert np.allclose(class_matrix2.sum(axis=1), class_vector_prime)
class_matrix0 = (class_matrix1 + class_matrix2) / 2
print(
    "Preliminary class matrix (mean of multiplication with tuple matrix and with inv tuple matrix)"
)
print(class_matrix0, class_matrix0.sum())
print()


# Transition probability Y=>Z given X and Y
triple0 = np.empty((NCLASSES, NCLASSES, NCLASSES))
for n in range(NCLASSES):
    curr = tuple0 * prop3d[n]
    triple0[n] = curr / curr.sum(axis=1)[:, None]

triple0_as_4d = np.zeros((NCLASSES, NCLASSES, NCLASSES, NCLASSES))
for x in range(NCLASSES):
    for y in range(NCLASSES):
        triple0_as_4d[x, y, y] = triple0[x, y]
triple0_wide = triple0_as_4d.reshape((NCLASSES * NCLASSES), (NCLASSES * NCLASSES))
class_matrix0_wide = class_matrix0.reshape(NCLASSES * NCLASSES)
class_matrix = adjust_class_vector(class_matrix0_wide, triple0_wide).reshape(
    (NCLASSES, NCLASSES)
)

print("Class matrix (adjusted)")
print(class_matrix, class_matrix.sum())
print()

print("Class matrix multiplied with triple matrix")
class_triple = class_matrix[:, :, None] * triple0
class_matrix_next = class_triple.sum(axis=0)
print(class_matrix_next, class_matrix_next.sum())
print(
    "Max class vector drift (from class matrix)",
    np.abs(class_matrix_next.sum(axis=1) - class_vector_prime).max(),
)
print("Max class matrix drift", np.abs(class_matrix_next - class_matrix).max())
print()
print("Final class vector")
class_vector_prime = class_matrix.sum(axis=1)
print(class_vector_prime)
print(
    "Final max class vector drift (from tuple0)",
    np.abs(
        (class_vector_prime[:, None] * tuple0).sum(axis=1) - class_vector_prime
    ).max(),
)

# Now we have class_vector_prime for class probabilities,
#  and class_matrix for class x class probabilities,
#  and class_triple for class x class x class probabilities
# all of which are (somewhat) stable against multiplication with the transition matrices
# We privilege the FORWARD direction for transition:
#   the class matrix multiplied with the FORWARD triple matrix, summed over the 3rd axis => class matrix

print(2, 100 * class_matrix)

levels = {2: (class_matrix, class_matrix)}

print()
# for level in range(3, 10):
for level in range(3, 5):  ###
    mins0, maxs0 = levels[level - 1]
    mins = (mins0[:, :, None] * triple0).min(axis=2)
    maxs = (maxs0[:, :, None] * triple0).max(axis=2)
    print(level)
    print(100 * mins)
    print(100 * maxs)
    print()
    levels[level] = mins, maxs


def split(word, size, level):
    words = [word + (i,) for i in range(NCLASSES)]

    if len(word) == 1:
        letter = word[0]
        vals = class_matrix[letter]
        mins = levels[level][0][letter]
        maxs = levels[level][1][letter]
        result = vals, mins, maxs
    elif len(word) == 2:
        vals = class_triple[word[0], word[1]]
        prev_mins, prev_maxs = levels[level - 1]
        mins3d = prev_mins[:, :, None] * triple0
        maxs3d = prev_maxs[:, :, None] * triple0
        mins = mins3d[word[0], word[1]]
        maxs = maxs3d[word[0], word[1]]
        if level == 3:
            assert np.allclose(mins, maxs)
            assert np.allclose(vals, mins)
        result = vals, mins, maxs
    else:
        deep = len(word) - 1
        letter1, letter2 = words[-2:]
        vals = class_matrix[letter2] * size
        mins = levels[level - deep][0][letter2] * size
        maxs = levels[level - deep][1][letter2] * size
        result = vals, mins, maxs

    return Tree(level, words, result[0], result[1], result[2])


from dataclasses import dataclass

from dataclasses import dataclass, field
from typing import List


@dataclass
class Tree:
    level: int
    words: List[tuple] = field(default_factory=list)
    sizes: List[float] = field(default_factory=list)
    mins: List[float] = field(default_factory=list)
    maxs: List[float] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.sizes, np.ndarray):
            self.sizes = self.sizes.tolist()
        if isinstance(self.mins, np.ndarray):
            self.mins = self.mins.tolist()
        if isinstance(self.maxs, np.ndarray):
            self.maxs = self.maxs.tolist()

    def print_words(self):
        for word, size, ccmin, ccmax in zip(
            self.words, self.sizes, self.mins, self.maxs
        ):
            w = "".join([str(i) for i in word])
            print(f"{w} size {100*size:.4f} min {100*ccmin:.4f} max {100*ccmax:.4f}")
        print()

    def _pop(self, index):
        self.words.pop(index)
        self.sizes.pop(index)
        self.mins.pop(index)
        self.maxs.pop(index)

    def _append(self, tree: "Tree"):
        self.words += tree.words
        self.sizes += tree.sizes
        self.mins += tree.mins
        self.maxs += tree.maxs

    def split(self, to_split):
        split_word = self.words[to_split]
        split_size = self.sizes[to_split]
        new_tree = split(split_word, split_size, self.level)
        assert abs(sum(new_tree.sizes) - split_size < 0.0001), (
            split_word,
            100 * sum(new_tree.sizes),
            100 * split_size,
        )
        assert max(new_tree.maxs) == self.maxs[to_split], (
            split_word,
            100 * max(new_tree.maxs),
            100 * self.maxs[to_split],
        )
        assert min(new_tree.mins) == self.mins[to_split]
        self._pop(to_split)
        self._append(new_tree)

    def sort_min(self):
        inds = np.argsort(self.mins)
        self.words[:] = [self.words[i] for i in inds]
        self.sizes[:] = [self.sizes[i] for i in inds]
        self.mins[:] = [self.mins[i] for i in inds]
        self.maxs[:] = [self.maxs[i] for i in inds]

    def cutoff(self, pos):
        self.words[:] = self.words[pos:]
        self.sizes[:] = self.sizes[pos:]
        self.mins[:] = self.mins[pos:]
        self.maxs[:] = self.maxs[pos:]


THRESHOLD = 0.01  # keep 99 %


def prune(level):
    words = [(i,) for i in range(NCLASSES)]
    mins0, maxs0 = levels[level]
    mins = mins0.min(axis=1)
    maxs = maxs0.max(axis=1)
    tree = Tree(
        level=level, words=words, sizes=class_vector_prime, mins=mins, maxs=maxs
    )
    tree.print_words()

    # We need a clean partition (no overlapping min/max)
    #  between words to eliminate and words to keep
    tree.sort_min()
    tree.print_words()

    while 1:
        discarded = 0
        pos = 0
        nwords = len(tree.words)
        threshold_cmpl = sum(tree.sizes) - THRESHOLD
        while 1:
            discarded += tree.sizes[pos]
            if discarded > THRESHOLD:
                break
            pos += 1
        if pos == 0:
            assert len(tree.words[pos]) < level, tree.words[pos]
            to_split = pos
        else:
            highest_to_discard = pos - 1
            assert (
                sum([tree.sizes[p] for p in range(highest_to_discard + 1)]) <= THRESHOLD
            )
            lowest_to_keep = highest_to_discard + 1
            assert sum([tree.sizes[p] for p in range(lowest_to_keep + 1)]) > THRESHOLD
            threshold1 = max(tree.maxs[: highest_to_discard + 1])
            threshold2 = tree.mins[lowest_to_keep]
            print(
                "Threshold1",
                100 * threshold1,
                tree.words[highest_to_discard],
                "Threshold2",
                100 * threshold2,
                tree.words[lowest_to_keep],
            )
            print()
            if threshold1 <= threshold2:
                if tree.mins[lowest_to_keep] == tree.maxs[lowest_to_keep]:
                    sum0 = sum(
                        [tree.sizes[p] for p in range(lowest_to_keep + 1, nwords)]
                    )
                    sum1 = sum([tree.sizes[p] for p in range(lowest_to_keep, nwords)])
                    assert sum0 < threshold_cmpl and sum1 >= threshold_cmpl, (
                        sum0,
                        sum(tree.sizes),
                        threshold_cmpl,
                        sum1,
                    )
                    final_threshold = tree.mins[lowest_to_keep]
                    break
                else:
                    to_split = lowest_to_keep
            else:
                to_split = np.argmax(tree.maxs[: highest_to_discard + 1])
        tree.split(to_split)
        print()
        tree.sort_min()
        tree.print_words()
        print()

    tree.sort_min()
    siz = 0
    for n in range(len(tree.words)):
        siz += tree.sizes[n]
        if siz > THRESHOLD:
            break
    else:
        raise AssertionError
    tree.cutoff(n)
    tree.print_words()
    assert sum(tree.sizes) > threshold_cmpl
    assert sum(tree.sizes[:-1]) < threshold_cmpl
    assert min(tree.mins) == min(tree.maxs) == final_threshold

    # Brute force validation
    accum = 0
    wordset = set(tree.words)

    def recurse(word, size, remaining_level):
        nonlocal accum
        if remaining_level == 0:
            for lev in range(1, level + 1):
                w = word[:lev]
                if w in wordset:
                    assert size >= final_threshold, (
                        word,
                        100 * size,
                        100 * final_threshold,
                    )
                    accum += size
                    break
            else:
                assert size < final_threshold, (word, 100 * size, 100 * final_threshold)
            return
        for cl in range(NCLASSES):
            new_word = word + (cl,)
            if len(word) == 0:
                new_size = class_vector_prime[cl]
            elif len(word) == 1:
                new_size = class_matrix[word[0], cl]
            else:
                partition = triple0[word[0], word[1]]
                assert np.isclose(partition.sum(), 1)
                new_size = size * partition[cl]
            recurse(new_word, new_size, remaining_level - 1)

    recurse((), 1, level)
    assert abs(accum - sum(tree.sizes)) < 0.001, (accum, sum(tree.sizes))
    # / validation


print("*" * 70)
print("Level 3")
prune(3)
# print("Level 4")
# prune(4)
