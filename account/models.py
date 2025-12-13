from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    phone = models.CharField(max_length=15, unique=True, null=False, blank=False)
    age = models.PositiveIntegerField(null=True, blank=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    user_type = models.CharField(max_length=20, 
        choices=[('assistant', 'assistant'), ('volunteer', 'volunteer'),
                ('blind','blind'),('deaf','deaf')], default='volunteer')
    gender = models.CharField(max_length=10,choices=[('male', 'male'),('female', 'female')] ,default='male')
    can_write = models.BooleanField(default=False,null=True, blank=True)
    can_speak_with_sign_language = models.BooleanField(default=False,null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    REQUIRED_FIELDS = ['age', 'address', 'phone', 'user_type', 'gender', 'can_write', 'can_speak_with_sign_language']

    def __str__(self):
        return f"{self.username} - {self.user_type}"    
class Post(models.Model):

    HELP_WAITING = 0
    IN_HELP_TRIP = 1
    FINISHED = 2

    STATE_CHOICES = [
        (HELP_WAITING, 'Help Waiting'),
        (IN_HELP_TRIP, 'In Help Trip'),
        (FINISHED, 'Finished Helped'),
    ]
    title = models.CharField(max_length=255)
    content = models.TextField()
    author = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='posts')
    state = models.IntegerField(choices=STATE_CHOICES, default=HELP_WAITING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
