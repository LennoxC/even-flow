import dataclasses
import typing
import yaml

def from_dict(cls, data):
    """
    Recursively build a dataclass from a dictionary.
    """

    if data is None:
        return None

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = typing.get_type_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        ftype = field_types[field.name]
        kwargs[field.name] = _convert(ftype, value)

    return cls(**kwargs)

def _convert(ftype, value):
    origin = typing.get_origin(ftype)

    # handle List[SomeDataclass]
    if origin in (list, typing.List):
        (inner_type,) = typing.get_args(ftype)
        return [_convert(inner_type, item) for item in value]

    # handle Optional[X] / Union[X, None]
    if origin is typing.Union:
        args = [a for a in typing.get_args(ftype) if a is not type(None)]
        if len(args) == 1:
            return _convert(args[0], value)
        return value

    # handle nested dataclasses
    if dataclasses.is_dataclass(ftype):
        return from_dict(ftype, value)

    # handle tuples specified as YAML lists, e.g. input_dim: [3, 32, 32]
    if origin is tuple:
        return tuple(value)

    # plain field
    return value

def load_yaml(cls, path):
    """
    Load a dataclass from a YAML file.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return from_dict(cls, data)