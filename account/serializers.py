from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import City, CustomUser, Post
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Post

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        token['username'] = user.username
        token['email'] = user.email
        token['user_type'] = user.user_type

        return token

    def validate(self, attrs):
        data = super().validate(attrs)

        user_data = {
            "id": self.user.id,
            "username": self.user.username,
            "email": self.user.email,
            "user_type": self.user.user_type,
        }

        # 🔹 إذا Assistant → رجّع المرضى (blind / deaf)
        patients = list(
            self.user.patients.filter(
                user_type__in=['blind', 'deaf']
            ).values('id', 'username', 'user_type')
        )

        user_data["patient"] = patients[0] if patients else None

        # 🔹 إذا Blind / Deaf → رجّع assistant
        if self.user.user_type in ['blind', 'deaf']:
            user_data["assistant_id"] = (
                self.user.assistant.id if self.user.assistant else None
            )

        data["user"] = user_data
        return data

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    assistant = serializers.IntegerField(required=False, allow_null=True)

    class Meta:
        model = CustomUser
        fields = [
            'username', 'email', 'phone', 'age', 'address',
            'gender', 'can_write', 'can_speak_with_sign_language',
            'is_active', 'user_type', 'assistant', 'password'
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
        fields = ['id', 'username', 'email', 'phone', 'age', 'address','gender', 'can_write','can_speak_with_sign_language'
                  ,'is_active', 'user_type']

class SmartVisionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
  

class CitySerializer(serializers.ModelSerializer):
    class Meta:
        model = City
        fields = ['id', 'name'] 
    def create(self, validated_data):
        city = City.objects.create(**validated_data)
        return city    
    

class PostSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)
    city = serializers.PrimaryKeyRelatedField(
        queryset=City.objects.all(),
        write_only=True
    )
    city_data = CitySerializer(source='city', read_only=True)

    class Meta:
        model = Post
        fields = [
            'id',
            'title',
            'content',
            'city',      # FK → client sends ID
            'city_data', 
            'author',    # FK → read-only
            'state',
            'created_at',
            'updated_at',
        ]

class STTRequestSerializer(serializers.Serializer):
    frames = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
