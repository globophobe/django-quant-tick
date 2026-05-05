from django.contrib import admin
from django.urls import path

from .production import urlpatterns

urlpatterns = [
    path("admin/", admin.site.urls),
    *urlpatterns,
]
