from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import *
from django.conf import settings
from django.conf.urls.static import static




router = DefaultRouter()
router.register('users', UserViewSet, basename='user')
router.register(r'cities', CityViewSet, basename='city')
router.register(r'posts', PostViewSet, basename='post')

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('delete-account/', DeleteOwnAccountView.as_view(), name='delete-account'),
    path('vision/', SmartVisionView.as_view(), name='smart-vision'),
    path("stt/", STTView.as_view(), name="stt"),
    path("sign-language/", SignLanguageView.as_view(), name="sign-language"),
    path("forgot-password/", ForgotPasswordRequestView.as_view(), name="forgot-password"),
    path("reset-password/", ResetPasswordConfirmView.as_view(), name="reset-password"),


]
urlpatterns += router.urls
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
