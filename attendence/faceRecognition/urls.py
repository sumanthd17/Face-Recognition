from django.conf.urls import url
from faceRecognition import views

app_name = 'faceRecognition'

urlpatterns = [
    #path('', views.index, name='index'),
    # /faceRecognition/
    url(r'^$', views.IndexView.as_view(), name='index'),

    # faceRecognition/session/entry
    url(r'^session/entry/$', views.SessionEntry.as_view(), name='session-entry'),

    # faceRecognition/session/id
    url(r'^session/(?P<pk>[0-9]+)/$', views.SessionUpdate.as_view(), name='session-update'),

    #faceRecognition/session/(?P<pk>[0-9]+)/delete
    url(r'^session/(?P<pk>[0-9]+)/delete$', views.SessionDelete.as_view(), name='session-delete'),

    # register user
    url(r'^register/', views.register),
]
