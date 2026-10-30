from ..public import get_public_json


def get_upbit_response(path, params):
    return get_public_json("https://api.upbit.com/v1", path, params, namespace="UPBIT")
