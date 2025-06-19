def encode_sex(sex: str) -> int:
    return 1 if sex == 'male' else 0


def encode_embarked(embarked: str) -> int:
    mapping = {'C': 0, 'Q': 1, 'S': 2}
    return mapping.get(embarked, -1)  # -1 if unknown


def encode_title(title: str) -> int:
    mapping = {'Master': 0, 'Miss': 2, 'Mr': 3, 'Mrs': 4}
    return mapping.get(title, 1)  # 1 = rare title


def bin_fare(fare: float) -> int:
    if 0 <= fare <= 7.91:
        return 0
    elif fare <= 14.454:
        return 3
    elif fare <= 31:
        return 1
    else:
        return 2


def bin_age(age: float) -> int:
    if age <= 16:
        return 0
    elif age <= 32:
        return 1
    elif age <= 48:
        return 2
    elif age <= 64:
        return 3
    else:
        return 4


def is_alone(familysize: int) -> int:
    return 1 if familysize == 1 else 0
