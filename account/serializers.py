from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import City, CustomUser, Post
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Post

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Optional: add claims into the token itself
        token['username'] = user.username
        token['email'] = user.email
        token['age'] = getattr(user, 'age', None)

        return token

    def validate(self, attrs):
        data = super().validate(attrs)

        # Add user info to response body
        data.update({
            "user": {
                "id": self.user.id,
                "username": self.user.username,
                "email": self.user.email,
                "user_type": self.user.user_type
            }
        })

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
        assistant = attrs.get('assistant')

        # Flutter يرسل 0 → نعتبره None
        if assistant == 0:
            attrs['assistant'] = None

        # blind / deaf لازم يكون لهم assistant
        if user_type in ['blind', 'deaf'] and not attrs.get('assistant'):
            raise serializers.ValidationError({
                "assistant": "Blind or deaf user must have an assistant."
            })

        # assistant لا يجب أن يكون له assistant
        if user_type == 'assistant':
            attrs['assistant'] = None

        return attrs

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = CustomUser.objects.create_user(**validated_data)
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
  
class PostSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Post
        fields = [
            'id',
            'title',
            'content',
            'city',      # FK → client sends ID
            'author',    # FK → read-only
            'state',
            'created_at',
            'updated_at',
        ]

class CitySerializer(serializers.ModelSerializer):
    class Meta:
        model = City
        fields = ['id', 'name'] 
    def create(self, validated_data):
        city = City.objects.create(**validated_data)
        return city    
    
class STTRequestSerializer(serializers.Serializer):
    frames = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=False
    )
