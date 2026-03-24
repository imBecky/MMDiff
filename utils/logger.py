class NoneDict(dict):
    """dict wrapper that returns None for missing keys."""

    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """Recursively convert mappings/lists into NoneDict containers."""
    if isinstance(opt, dict):
        out = NoneDict()
        for key, value in opt.items():
            out[key] = dict_to_nonedict(value)
        return out
    if isinstance(opt, list):
        return [dict_to_nonedict(item) for item in opt]
    if isinstance(opt, tuple):
        return tuple(dict_to_nonedict(item) for item in opt)
    return opt
