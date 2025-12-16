from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import CustomUser, Post
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


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
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'phone', 'age', 'address','gender', 'can_write','can_speak_with_sign_language'
                  ,'is_active', 'user_type', 'password']


    """def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Passwords do not match."})
        return attrs"""

    def create(self, validated_data):
        user = CustomUser.objects.create_user(**validated_data)
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['id', 'username', 'email', 'phone', 'age', 'address','gender', 'can_write','can_speak_with_sign_language'
                  ,'is_active', 'user_type']


class SmartVisionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
  
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'author','state', 'created_at', 'updated_at']
    def create(self, validated_data):
        post = Post.objects.create(**validated_data)
        return post