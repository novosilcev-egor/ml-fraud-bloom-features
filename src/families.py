def expand_family(spec: dict) -> list[str]:
    prefix = spec["prefix"]
    lo, hi = spec["range"]
    zero_pad = spec.get("zero_pad")
    if zero_pad:
        return [f"{prefix}{str(i).zfill(zero_pad)}" for i in range(lo, hi + 1)]
    return [f"{prefix}{i}" for i in range(lo, hi + 1)]


def expand_feature_list(entries: list, config: dict) -> list[str]:
    families = config["features"]["families"]
    columns = []
    for entry in entries:
        if isinstance(entry, dict) and "family" in entry:
            columns.extend(expand_family(families[entry["family"]]))
        else:
            columns.append(entry)
    return columns


def get_forbidden_columns(config: dict) -> list[str]:
    return config["features"]["forbidden_as_features"]


def get_d_columns(config: dict) -> list[str]:
    return list(config["d_features"]["resolution"].keys())


def get_active_d_columns(config: dict) -> list[str]:
    resolution = config["d_features"]["resolution"]
    return [col for col, res in resolution.items() if res != "excluded"]


def candidate_feature_columns(config: dict) -> list[str]:
    features = config["features"]
    columns = []
    columns.extend(expand_feature_list(features["numeric"], config))
    columns.extend(expand_feature_list(features["categorical"], config))
    columns.extend(expand_feature_list(features.get("numeric_extra", []), config))
    for col in get_active_d_columns(config):
        columns.append(col)
        columns.append(col + "Norm")
    return columns
