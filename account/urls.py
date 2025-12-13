from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import *

router = DefaultRouter()
router.register('users', UserViewSet, basename='user')

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('delete-account/', DeleteOwnAccountView.as_view(), name='delete-account'),
    path('vision/', SmartVisionView.as_view(), name='smart-vision'),
    path('posts/', PostViewSet.as_view({'get': 'list'}), name='post-list'),
    path('post/create/', CreatePostView.as_view(), name='post-create'),

]

urlpatterns += router.urls
