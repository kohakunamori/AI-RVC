"""Keep Numba's cached audio kernels backed by real, relocatable source files."""

# Merely adding .py files as data leaves PYZ modules taking precedence, with
# relative co_filename values. That prevents Numba's configured/in-tree cache
# locators from finding the source when launched outside the source checkout.
# Use source imports, including librosa/resampy kernels reached during mixing.
module_collection_mode = {
    "lib.mixer": "py",
    "librosa": "py",
    "resampy": "py",
}
