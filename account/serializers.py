from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import City, CustomUser, Post
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Post


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    device_token = serializers.CharField(
        required=False,
        allow_blank=True
    )
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        token['email'] = user.email
        token['user_type'] = user.user_type
        token['in_home'] = user.in_home
        token['current_location'] = user.current_location
        return token

    def validate(self, attrs):
        device_token = attrs.pop("device_token", None)
        data = super().validate(attrs)
        user = self.user

        if device_token:
            user.device_token = device_token
            user.save(update_fields=["device_token"])
        
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "user_type": user.user_type,
        }

        # ✅ Assistant → رجّع patient واحد فقط
        if user.user_type == 'assistant':
            patient = (
                user.patients
                .filter(user_type__in=['blind', 'deaf'])
                .values('id', 'username', 'email', 'user_type','in_home','current_location')
                .first()
            )
            user_data["patient"] = patient

        # ✅ Blind / Deaf → رجّع assistant object
        if user.user_type in ['blind', 'deaf']:
            user_data["assistant"] = (
                {
                    "id": user.assistant.id,
                    "username": user.assistant.username,
                    "email": user.assistant.email,
                    "user_type": user.assistant.user_type,
                }
                if user.assistant else None
            )

        data["user"] = user_data
        return data


#user registration serializer
class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    assistant = serializers.IntegerField(required=False, allow_null=True)

    class Meta:
        model = CustomUser
        fields = [
            'username', 'email', 'phone', 'age', 'address',
            'gender', 'can_write', 'can_speak_with_sign_language',
            'is_active', 'user_type', 'assistant', 'password','device_token'
        ]

    def validate(self, attrs):
        user_type = attrs.get('user_type')
        assistant_id = attrs.get('assistant')

        # Flutter يرسل 0 → None
        if assistant_id in [0, '0', None]:
            attrs['assistant'] = None
            assistant = None
        else:
            try:
                assistant = CustomUser.objects.get(
                    id=assistant_id,
                    user_type='assistant'
                )
                attrs['assistant'] = assistant
            except CustomUser.DoesNotExist:
                raise serializers.ValidationError({
                    "assistant": "Assistant not found."
                })

        # blind / deaf لازم assistant
        if user_type in ['blind', 'deaf'] and not attrs.get('assistant'):
            raise serializers.ValidationError({
                "assistant": "Blind or deaf user must have an assistant."
            })

        # assistant لا يكون له assistant
        if user_type == 'assistant':
            attrs['assistant'] = None

        return attrs


    def create(self, validated_data):
        password = validated_data.pop('password')
        assistant = validated_data.pop('assistant', None)

        user = CustomUser(**validated_data)
        user.assistant = assistant
        user.set_password(password)
        user.save()

        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'phone', 'age', 'address','current_location','in_home','gender', 'can_write','can_speak_with_sign_language'
                  ,'is_active', 'user_type','device_token']




#reset pass

class ForgotPasswordRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()

class ResetPasswordConfirmSerializer(serializers.Serializer):
    email = serializers.EmailField()
    #code = serializers.CharField(max_length=6)
    new_password = serializers.CharField(min_length=8, write_only=True)





class SmartVisionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
  
#city serializer
class CitySerializer(serializers.ModelSerializer):
    class Meta:
        model = City
        fields = ['id', 'name'] 
    def create(self, validated_data):
        city = City.objects.create(**validated_data)
        return city    
    
#post serializer
class PostSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    city = serializers.PrimaryKeyRelatedField(
        queryset=City.objects.all(),
        write_only=True
    )
    city_data = CitySerializer(source='city', read_only=True)

    volunteer = UserSerializer(read_only=True)
    

    help_requesters = UserSerializer(many=True, read_only=True)
    volunteer_id = serializers.PrimaryKeyRelatedField(
        source='volunteer',
        queryset=CustomUser.objects.filter(user_type='volunteer'),
        write_only=True,
        required=False,
        allow_null=True
    )

    help_requesters_ids = serializers.PrimaryKeyRelatedField(
        source='help_requesters',
        queryset=CustomUser.objects.filter(user_type='volunteer'),
        many=True,
        write_only=True,
        required=False,
        allow_empty=True
    )
    class Meta:
        model = Post
        fields = [
            'id',
            'title',
            'content',
            'city',      
            'city_data', 
            'author',    
            'volunteer',       
            'volunteer_id',    
            'help_requesters', 
            'help_requesters_ids',
            'state',
            'created_at',
            'updated_at',
        ]
        def validate_volunteer_id(self, value):
            if value == 0:
                return None
            return value

        def validate_help_requesters_ids(self, value):
            if isinstance(value, list):
                return [v for v in value if v != 0]
            return value


class STTRequestSerializer(serializers.Serializer):
    frames = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
class AssistantMiniSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'user_type']
class SignLanguageRequestSerializer(serializers.Serializer):
    input_type = serializers.ChoiceField(
        choices=["text", "audio"],
        help_text="Input type: text or base64 audio"
    )
    input_data = serializers.CharField(
        help_text="Arabic text OR base64-encoded audio"
    )

class SignLanguageResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    recognized_text = serializers.CharField(allow_null=True)
    video_base64 = serializers.CharField(allow_null=True)
    missing_words = serializers.ListField(child=serializers.CharField())
    found_matches = serializers.ListField()
    total_signs = serializers.IntegerField(required=False)
    error = serializers.CharField(required=False)
