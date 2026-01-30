import sys
import numpy as np


THRESHOLD = 0.10

class_vector = np.load("class_vector.npy")
NCLASSES = len(class_vector)
prop2d = np.load("propensity_2d.npy")
assert prop2d.shape == (NCLASSES, NCLASSES)
prop3d = np.load("propensity_3d.npy")
assert prop3d.shape == (NCLASSES, NCLASSES, NCLASSES)

conf_classes = []
for lib in ("AA", "AC", "CA", "CC"):
    conf_classes.append(np.load(f"conformer-classes-{lib}.npy"))
conf_classes = np.concatenate(conf_classes)
classes0, class_counts = np.unique(conf_classes, return_counts=True)
assert len(classes0) == NCLASSES and np.all(np.equal(classes0, np.arange(NCLASSES)))
conf_class_frac = class_counts / len(conf_classes)
print("Conformer class fractions:", conf_class_frac.tolist())
log_conf_class_frac = np.log(conf_class_frac)


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

log_class_matrix = np.log(class_matrix)
log_triple0 = np.log(triple0)

try:
    from scipy.special import logcumsumexp
except ImportError:

    def logcumsumexp(logx):
        m = np.maximum.accumulate(logx)
        return m + np.log(np.cumsum(np.exp(logx - m)))


try:
    from scipy.special import logsumexp
except ImportError:

    def logsumexp(logx):
        return np.max(logx) + np.log(np.sum(np.exp(logx - np.max(logx))))


def get_log_frac(log_class_frac, mask):
    f = log_class_frac.reshape(-1)
    m = mask.reshape(-1)
    return logsumexp(f[m])


def prune(log_matrix):
    level = log_matrix.ndim
    nr_rare = log_conf_class_frac  # test
    nr_rare_matrix = nr_rare
    for _ in range(level):
        nr_rare_matrix = nr_rare_matrix[..., None] + nr_rare.reshape(
            (1,) * (level - 1) + (-1,)
        )
    nr_rare_matrix_flat = nr_rare_matrix.reshape(-1).astype(np.float32)
    log_matrix_flat = log_matrix.reshape(-1).astype(np.float32)

    print("sort1", file=sys.stderr)
    sorting1 = log_matrix_flat.argsort().astype(np.int32)
    print("sort2", file=sys.stderr)
    sorting2 = (-nr_rare_matrix_flat[sorting1]).argsort(stable=True).astype(np.int32)
    sorting = sorting1[sorting2]
    del sorting1, sorting2
    print("sorted", file=sys.stderr)

    log_sorted = log_matrix_flat[sorting]
    cumsum_log_sorted = logcumsumexp(log_sorted)
    pos = np.searchsorted(cumsum_log_sorted, np.log(THRESHOLD))
    if pos == 0:
        print(f"Level {level}, no pruning")
    mask = np.zeros(len(log_matrix_flat), bool)
    mask[sorting[pos:]] = 1
    return pos, mask.reshape(log_matrix.shape)


log_conf_class_frac_2d = log_conf_class_frac[:, None] + log_conf_class_frac[None, :]

print()
pruned, mask = prune(log_class_matrix)
level = 2
print("Level", level)
print(
    f"Pruned combinations: {pruned}/{len(log_class_matrix.reshape(-1))} {100*pruned/len(log_class_matrix.reshape(-1)):.1f} %"
)
print(
    "Fragment pruning (close to threshold)",
    100 * np.exp(get_log_frac(log_class_matrix, mask)),
)
log_conf_frac = get_log_frac(log_conf_class_frac_2d, mask)
print(
    f"Log conformation pruning {log_conf_frac:.5f}, per level {log_conf_frac/level:.5f}"
)
print(
    f"Conformation pruning {np.exp(log_conf_frac):.5f}, per level {np.exp(log_conf_frac/level):.5f}"
)
print()

log_matrix = log_class_matrix
log_conf_class_frac_nd = log_conf_class_frac_2d
for level in range(3, 12 + 1):
    log_matrix = log_matrix[..., :, :, None] + log_triple0

    log_conf_class_frac_nd = log_conf_class_frac_nd[
        ..., None
    ] + log_conf_class_frac.reshape((1,) * (level - 1) + (-1,))
    pruned, mask = prune(log_matrix)

    print("Level", level)
    print(
        f"Pruned combinations: {pruned}/{len(log_matrix.reshape(-1))} {100*pruned/len(log_matrix.reshape(-1)):.1f} %"
    )
    print(
        "Fragment pruning (close to threshold)",
        100 * np.exp(get_log_frac(log_matrix, mask)),
    )
    log_conf_frac = get_log_frac(log_conf_class_frac_nd, mask)
    print(
        f"Log conformation pruning {log_conf_frac:.5f}, per level {log_conf_frac/level:.5f}"
    )
    print(
        f"Conformation pruning {np.exp(log_conf_frac):.5f}, per level {np.exp(log_conf_frac/level):.5f}"
    )
    print()

"""
RESULTS: 

Threshold = 1 %  (of the TOTAL, not per level!)

level 7-8: reduce 10 % (keep 90 %) per level
level 10-11: reduce 16 % per level
level 12: reduce 19 % per level  (92 % in total)
Underwhelming

Threshold = 5 %
level 5-10: reduce 35 %  per level
level 11-12: reduce 45 % per level, log(reduction)=-7.5 for level=12

Threshold = 10 %
Level 10: reduce 60 % per level
Level 11-12: reduce 65 % per level, log(reduction)=-13 for level=12


"""
