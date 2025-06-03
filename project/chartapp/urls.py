from django.urls import path
from . import views

urlpatterns = [
       path('', views.index, name='index'),  
   ]
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # your app URLs
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
