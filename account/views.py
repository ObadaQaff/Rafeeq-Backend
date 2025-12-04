from rest_framework import generics, permissions, viewsets
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .models import CustomUser
from .serializers import RegisterSerializer, UserSerializer
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from service.blind import SmartVisionSystem


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
    



class SmartVisionView(APIView):
  # You can change to AllowAny

    def post(self, request, *args, **kwargs):
        from .serializers import SmartVisionRequestSerializer

        serializer = SmartVisionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            system = SmartVisionSystem(credentials_json_string=data["credentials"])

            audio_file = system.process_image_from_flutter(
                base64_image=data["image"],
                conf_threshold=data["conf_threshold"],
                enable_ocr=data["enable_ocr"],
                force_announce=data["force_announce"],
            )

            if audio_file is None:
                return Response(
                    {"success": False, "message": "No objects detected."},
                    status=status.HTTP_200_OK
                )

            return Response(
                {
                    "success": True,
                    "audio_file": audio_file,
                    "message": "Audio generated successfully"
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {"success": False, "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
