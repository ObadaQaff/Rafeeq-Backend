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
from .serializers import CustomTokenObtainPairSerializer



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
    permission_classes = [permissions.IsAuthenticated]

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
    

#---------------
# Post View
#---------------
from .serializers import PostSerializer
from .models import Post
class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer



class CreatePostView(generics.CreateAPIView):
    serializer_class = PostSerializer

    def post(self, request, *args, **kwargs):
        serializer = PostSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



#----------------
# Smart Vision View
#----------------
class SmartVisionView(APIView):
    parser_classes = [MultiPartParser, FormParser]


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

