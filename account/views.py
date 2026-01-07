from rest_framework import generics, permissions, viewsets
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .models import CustomUser
from .serializers import RegisterSerializer, SmartVisionRequestSerializer, UserSerializer
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from service.blind import SmartVisionSystem
from drf_yasg.utils import swagger_auto_schema
from rest_framework.parsers import JSONParser
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import *
from .models import Post
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import City
from .serializers import CitySerializer
import base64
from .serializers import STTRequestSerializer
from service.STT.STT import ASLTranslatorFinal

#---------------
# User View
#---------------
class CustomLoginView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

class RegisterView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer

    def create(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            user = serializer.save()

            return Response({
                "Isuccess": True,
                "message": "User registered successfully",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "phone": user.phone,
                    "user_type": user.user_type,
                    "gender": user.gender,
                    "can_write": user.can_write,
                    "can_speak_with_sign_language": user.can_speak_with_sign_language
                }
            }, status=status.HTTP_201_CREATED)

        except ValidationError as e:
            # Serializer validation errors
            return Response({
                "Isuccess": False,
                "errors": e.detail
            }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # Unexpected server errors
            return Response({
                "Isuccess": False,
                "message": "An unexpected error occurred.",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# View / edit users (admin-only)
class UserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

# Custom logout endpoint
class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data["refresh"]
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({"success": "Logged out successfully"}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=400)

class DeleteOwnAccountView(generics.DestroyAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user
    
#----------------
# City ViewSet
#----------------
class CityViewSet(viewsets.ModelViewSet):
    queryset = City.objects.all()
    serializer_class = CitySerializer
    permission_classes = [permissions.IsAuthenticated]

    swagger_tags = ["Cities"]

    def create(self, request, *args, **kwargs):
            
        return super().create(request, *args, **kwargs)

    def update(self, request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response(
                {"detail": "Only admin users can update cities."},
                status=status.HTTP_403_FORBIDDEN
            )
        return super().update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response(
                {"detail": "Only admin users can delete cities."},
                status=status.HTTP_403_FORBIDDEN
            )
        return super().destroy(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_summary="List cities",
        operation_description="Public endpoint to list all cities"
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

#----------------
# Post ViewSet
#----------------
class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [AllowAny]

    swagger_tags = ["Posts"]

    def create(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return Response(
                {"detail": "Authentication required to create a post."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.save(author=request.user)

        return Response(serializer.data, status=status.HTTP_201_CREATED)


#----------------
# Smart Vision View
#----------------
class SmartVisionView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    #Api google 
    dic = {
}

    @swagger_auto_schema(
        request_body=SmartVisionRequestSerializer,
        responses={200: "Success", 400: "Bad Request"}
    )
    def post(self, request, *args, **kwargs):
    
        serializer = SmartVisionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            system = SmartVisionSystem(
                credentials_json_string=self.dic
            )

            audio_file = system.process_image_from_flutter(
                base64_image=data["image"],
                conf_threshold=0.5,
                enable_ocr=True,
                force_announce=False,
            )

            if not audio_file:
                return Response(
                    {"success": False, "message": "No objects detected"},
                    status=200
                )

            return Response(
                {"success": True, "audio_file": audio_file},
                status=200
            )

        except Exception as e:
            return Response(
                {"success": False, "error": str(e)},
                status=500
            )

#----------------
# STT ViewSet
#----------------
class STTView(APIView):
    permission_classes = []  # AllowAny
    @swagger_auto_schema(
        request_body=STTRequestSerializer,
        responses={200: "Success"}
    )
    def post(self, request):
        serializer = STTRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        frames_base64 = serializer.validated_data['frames']

        # Decode base64 → bytes
        frames_bytes = []
        for frame in frames_base64:
            try:
                frames_bytes.append(base64.b64decode(frame))
            except Exception:
                return Response(
                    {"success": False, "error": "Invalid base64 frame"},
                    status=400
                )

        translator = ASLTranslatorFinal()
        result = translator.process_frames_from_flutter(frames_bytes)

        if not result['success']:
            return Response(result, status=400)

        # Convert bytes → base64 for Flutter
        response = {
            "success": True,
            "has_audio": result["has_audio"],
            "has_text": result["has_text"],
            "audio_file": (
                base64.b64encode(result["audio_file"]).decode()
                if result["audio_file"] else None
            ),
            "text_file": (
                result["text_file"].decode("utf-8")
                if result["text_file"] else None
            )
        }

        return Response(response, status=200)