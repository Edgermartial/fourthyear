from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('about_us/', views.about_us, name='about_us'),
    path('blog_news/', views.blog_news, name='blog_news'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('help-support/', views.help_support, name='help_support'),
    path('community_forum/', views.community_forum, name='community_forum'),
    path('adaptation_strategies/', views.adaptation_strategies, name='adaptation_strategies'),
    path('crop_rec/', views.crop_rec, name='crop_rec'),

    # Authentication (clean URLs)
    path('accounts/register/', views.register_view, name='register'),
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),

    # API Endpoints
    path('predict_weather/', views.predict_weather, name='predict_weather'),
    path('recommend/', views.recommend_crop, name='recommend_crop'),
    path('predict_time/', views.predict_time, name='predict_time'),
    path('api/send-message/', views.receive_message, name='send_message'),
]
