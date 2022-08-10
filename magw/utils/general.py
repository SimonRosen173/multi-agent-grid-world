from typing import Optional, Dict


def copy_to_dict(from_dict: Optional[Dict], to_dict: Dict):
    if from_dict is None:
        from_dict = {}
    for key in from_dict.keys():
        to_dict[key] = from_dict[key]
    return to_dict