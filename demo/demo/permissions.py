import logging

from decouple import config
from google.oauth2 import id_token
from rest_framework.permissions import BasePermission
from rest_framework.request import Request

logger = logging.getLogger(__name__)


class GCPServicePermission(BasePermission):
    def has_permission(self, request: Request, *args, **kwargs) -> bool:
        """Has permission."""
        header = request.headers.get("Authorization")
        if header:
            auth_type, token = header.split(" ", 1)
            if auth_type.lower() == "bearer":
                try:
                    id_token.verify_oauth2_token(
                        token, request, config("OAUTH2_AUDIENCE")
                    )
                except ValueError:
                    logger.info(f"Unauthorized: {token}")
                else:
                    return True
