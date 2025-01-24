from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.Index, name='index'),
    path('process_video/', views.process_video, name='process_video'),
    path('train-yourself/', views.train_yourself, name='train_yourself'),
    path('train_yourself_video/', views.train_yourself_video, name='train_yourself_video'),
    path('elbowstrike/', views.elbowstrike, name='elbowstrike'),
    path('upload/', views.upload_video, name='upload_video'),
    path('safemap/', views.safemap, name='safemap'),
    path('defendchat/', views.defendchat, name='defendchat'),
]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
