from rest_framework.permissions import BasePermission


class AllowAnyInternal(BasePermission):
    """
    Placeholder permission class for internal services.
    """

    def has_permission(self, request, view):
        return True
