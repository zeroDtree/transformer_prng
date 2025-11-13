from .MT19937 import (
    load_mt19937,
    load_mt19937_8bits,
    load_mt19937_8bits_with_eval,
    load_mt19937_12bits,
    load_mt19937_12bits_with_eval,
    load_mt19937_16bits,
    load_mt19937_16bits_with_eval,
    load_mt19937_32bits,
    load_mt19937_32bits_with_eval,
)

# return (train_set, validation_set, test_set)
DATASETMAPPING = {
    # language dataset ===========================================
    "mt19937": load_mt19937,
    "mt19937-8": load_mt19937_8bits,
    "mt19937-12": load_mt19937_12bits,
    "mt19937-16": load_mt19937_16bits,
    "mt19937-32": load_mt19937_32bits,
    "mt19937-8-eval": load_mt19937_8bits_with_eval,
    "mt19937-12-eval": load_mt19937_12bits_with_eval,
    "mt19937-16-eval": load_mt19937_16bits_with_eval,
    "mt19937-32-eval": load_mt19937_32bits_with_eval,
}
