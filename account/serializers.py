from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import CustomUser

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
  
