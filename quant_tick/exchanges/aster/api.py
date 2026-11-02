from ..public import get_public_json


def get_aster_response(path, params):
    return get_public_json(
        "https://fapi.asterdex.com/fapi/v3", path, params, namespace="ASTER"
    )
