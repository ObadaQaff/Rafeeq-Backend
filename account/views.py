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
from rest_framework.decorators import action
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import City
from .serializers import CitySerializer
import base64
from .serializers import STTRequestSerializer
from service.STT.STT import ASLTranslatorFinal
from .serializers import SignLanguageRequestSerializer, SignLanguageResponseSerializer
from service.TTS.TTS import SignLanguageVideoGenerator
import os
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
                    "can_speak_with_sign_language": user.can_speak_with_sign_language,
                    "assistant": user.assistant.id if user.assistant else None
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



import random
from django.utils import timezone
from django.core.mail import send_mail
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.contrib.auth import get_user_model

from .models import PasswordResetCode
from .serializers import ForgotPasswordRequestSerializer, ResetPasswordConfirmSerializer

User = get_user_model()


class ForgotPasswordRequestView(APIView):
    permission_classes = [permissions.AllowAny]
    @swagger_auto_schema(
        request_body=ForgotPasswordRequestSerializer,
        responses={200: "Success", 400: "Bad Request"}
    )
    def post(self, request):
        serializer = ForgotPasswordRequestSerializer(data=request.data)

        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"].lower().strip()

        # Always return success (prevents email enumeration)
        # But only send email if user exists.
        user = User.objects.filter(email__iexact=email).first()
        if user:
            code = f"{random.randint(0, 999999):06d}"

            # invalidate old codes for this email
            PasswordResetCode.objects.filter(email__iexact=email, used=False).update(used=True)

            PasswordResetCode.objects.create(email=email, code=code)

            send_mail(
                subject="Password reset code",
                message=f"Your password reset code is: {code}\nThis code expires in 10 minutes.",
                from_email=None,   # uses DEFAULT_FROM_EMAIL
                recipient_list=[email],
                fail_silently=False,
            )

        return Response(
            {"success": True, "message": "If the email exists, a reset code was sent."},
            status=status.HTTP_200_OK
        )


class ResetPasswordConfirmView(APIView):
    permission_classes = [permissions.AllowAny]
    @swagger_auto_schema(
        request_body=ResetPasswordConfirmSerializer,
        responses={200: "Success", 400: "Bad Request"}
    )

    def post(self, request):
        serializer = ResetPasswordConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"].lower().strip()
        # code = serializer.validated_data["code"].strip()
        new_password = serializer.validated_data["new_password"]

        user = CustomUser.objects.filter(email__iexact=email).first()
        if not user:
            return Response({"success": False, "message": "Invalid code."}, status=status.HTTP_400_BAD_REQUEST)

        # prc = (
        #     PasswordResetCode.objects
        #     .filter(email__iexact=email, used=False)
        #     .order_by("-created_at")
        #     .first()
        # )

        # if not prc or prc.is_expired():
        #     return Response({"success": False, "message": "Invalid or expired code."}, status=status.HTTP_400_BAD_REQUEST)

        # # mark used
        # prc.used = True
        # prc.save(update_fields=["used"])

        # reset password
        user.set_password(new_password)
        user.save(update_fields=["password"])

        return Response({"success": True, "message": "Password updated successfully."}, status=status.HTTP_200_OK)



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
    permission_classes = [permissions.IsAuthenticated]

    swagger_tags = ["Posts"]


    @action(
        detail=True,
        methods=["post"],
        url_path="request-help"
    )
    def request_help(self, request, pk=None):
        post = self.get_object()
        user = request.user

        # ❌ Author cannot request help
        if user == post.author:
            return Response(
                {"detail": "Author cannot request help"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ❌ Only volunteers can request help
        if user.user_type != "volunteer":
            return Response(
                {"detail": "Only volunteers can request help"},
                status=status.HTTP_403_FORBIDDEN
            )

        # ❌ Already requested
        if post.help_requesters.filter(id=user.id).exists():
            return Response(
                {"detail": "You already requested to help"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ❌ Volunteer already chosen
        if post.volunteer:
            return Response(
                {"detail": "Volunteer already selected"},
                status=status.HTTP_400_BAD_REQUEST
            )

        post.help_requesters.add(user)

        return Response(
            {"detail": "Request to help added successfully"},
            status=status.HTTP_200_OK
        )

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

    @swagger_auto_schema(
        request_body=SmartVisionRequestSerializer,
        responses={200: "Success", 400: "Bad Request"}
    )
    def post(self, request, *args, **kwargs):
    
        serializer = SmartVisionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            system = SmartVisionSystem()

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


from django.http import FileResponse, HttpResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import time
import os

class STTView(APIView):
    permission_classes = []  # AllowAny

    @swagger_auto_schema(
        request_body=STTRequestSerializer,
        responses={200: STTRequestSerializer, 400: "Bad Request"}
    )
    def post(self, request):
        serializer = STTRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        frames_base64 = serializer.validated_data['frames']

        frames_bytes = []
        for i, frame in enumerate(frames_base64):
            try:
                frames_bytes.append(base64.b64decode(frame))
            except Exception as e:
                return Response(
                    {"success": False, "error": f"Invalid base64 at frame {i}: {str(e)}"},
                    status=400
                )

        translator = ASLTranslatorFinal()
        result = translator.process_frames_from_flutter(frames_bytes)

        if not result.get('success', False):
            return Response(result, status=400)
        audio_file_bytes = result["audio_file"]
        audio_file_bytes = result.get("audio_file")

        if audio_file_bytes:
            timestamp = int(time.time() * 1000)
            filename = f"stt_audio_{timestamp}.mp3"

            # ✅ احفظ داخل MEDIA_ROOT
            relative_path = os.path.join("stt", filename)  # folder inside media
            saved_path = default_storage.save(relative_path, ContentFile(audio_file_bytes))

            # ✅ رجّع URL مثل SmartVision
            audio_url = settings.MEDIA_URL + saved_path.replace("\\", "/")

            return Response(
                {"success": True, "audio_file": audio_url},
                status=200
            )

        return Response({"success": False, "error": "No audio generated"}, status=400)

                # response = {
                #     "success": True,
                #     "has_audio": result.get("has_audio", False),
                #     "has_text": result.get("has_text", False),
                #     "audio_file": base64.b64encode(result["audio_file"]).decode() if result.get("audio_file") else None,
                #     "text_file": result["text_file"].decode("utf-8") if result.get("text_file") else None,
                # }

                # return Response(response, status=200)




from pathlib import Path
import os

# views.py is: /Users/obadaqafisheh/Rafeeq/Rafeeq/account/views.py
BASE_DIR = Path(__file__).resolve().parents[1]          # /Users/obadaqafisheh/Rafeeq/Rafeeq
SIGNS_DIR = BASE_DIR / "service" / "TTS" / "signs"
signs_dict_path = BASE_DIR / "service" / "TTS" / "signs_dictionary.json"

SIGN_GENERATOR = SignLanguageVideoGenerator(
    signs_dict_path=str(signs_dict_path),
    signs_dir=str(SIGNS_DIR),   # ✅ works on local + server
)

from django.conf import settings
from pathlib import Path

class SignLanguageView(APIView):
    permission_classes = []
    
    @swagger_auto_schema(
        request_body=SignLanguageRequestSerializer,
        responses={200: SignLanguageResponseSerializer, 400: "Bad Request"}
    )
    def post(self, request):
        serializer = SignLanguageRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        input_type = serializer.validated_data["input_type"]
        input_data = serializer.validated_data["input_data"]

        # ✅ where generated videos will be saved
        out_dir = Path(settings.MEDIA_ROOT) / "generated_signs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ✅ build base url correctly (works local + server)
        base_url = request.build_absolute_uri("/")[:-1]  # "http://host:port"

        result = SIGN_GENERATOR.process_from_flutter(
            input_data=input_data,
            input_type=input_type,
            output_dir=str(out_dir),
            base_url=base_url
        )

        if not result.get("success"):
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        # ✅ return small JSON with URL
        return Response({
            "success": True,
            "recognized_text": result.get("recognized_text"),
            "video_url": result.get("video_url"),
            "missing_words": result.get("missing_words", []),
            "found_matches": result.get("found_matches", []),
            "total_signs": result.get("total_signs", 0),
        }, status=status.HTTP_200_OK)






#notification with firebase
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.contrib.auth import get_user_model
from drf_yasg.utils import swagger_auto_schema

from .serializers import SendNotificationSerializer
from .firebase import send_fcm_to_token

User = get_user_model()

class SendNotificationByUserIdView(APIView):
    permission_classes = [permissions.AllowAny]  # change to IsAuthenticated later

    @swagger_auto_schema(
        request_body=SendNotificationSerializer,
        responses={200: "Success", 400: "Bad Request", 404: "User Not Found", 500: "Server Error"}
    )
    def post(self, request):
        serializer = SendNotificationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user_id = serializer.validated_data["user_id"]
        title = serializer.validated_data["title"]
        body = serializer.validated_data["body"]
        data = serializer.validated_data.get("data") or {}
        android_channel_id = serializer.validated_data.get("android_channel_id") or None

        user = User.objects.filter(id=user_id).first()
        if not user:
            return Response({"success": False, "error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        token = (user.device_token or "").strip()
        if not token:
            return Response({"success": False, "error": "User has no device_token."}, status=status.HTTP_400_BAD_REQUEST)

        # ✅ Detect call mode from data["type"]
        is_call = (data.get("type") == "incoming_call")
        ttl_seconds = None

        if is_call:
            # TTL usually 30 seconds for incoming calls
            try:
                ttl_seconds = int(data.get("ttl", 30))
            except (TypeError, ValueError):
                ttl_seconds = 30

            # default channel for calls
            if not android_channel_id:
                android_channel_id = "calls"

        try:
            message_id = send_fcm_to_token(
                token=token,
                title=title,
                body=body,
                data=data,
                android_channel_id=android_channel_id,
                is_call=is_call,
                ttl_seconds=ttl_seconds,
            )
            return Response({"success": True, "message_id": message_id}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
