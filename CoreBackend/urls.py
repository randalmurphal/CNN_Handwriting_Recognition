from django.urls import path

from . import views

urlpatterns = [
    path('', views.AboutUsAndDownload, name='index'),
    path('upload/', views.UploadPage, name='upload'),
    path('results/', views.EDAAndResults, name='edaresults'),
    path('contributors/',views.Contributors, name='contributors'),
    path('project/',views.Project, name='project'),
]