from django.contrib.auth.models import AbstractUser
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from datetime import timedelta




class CustomUser(AbstractUser):
    phone = models.CharField(max_length=15, unique=True)
    age = models.PositiveIntegerField(null=True, blank=True)

    #Address and location fields
    address = models.CharField(max_length=1000, blank=True, null=True)
    current_location = models.CharField(max_length=1000, blank=True, null=True)
    in_home = models.BooleanField(default=True , blank=True, null=True)


    device_token = models.CharField(max_length=1000, blank=True, null=True)
    USER_TYPES = [
        ('assistant', 'assistant'),
        ('blind', 'blind'),
        ('deaf', 'deaf'),
        ('volunteer', 'volunteer'),
    ]

    user_type = models.CharField(
        max_length=20,
        choices=USER_TYPES,
        default='volunteer'
    )

    gender = models.CharField(
        max_length=10,
        choices=[('male', 'male'), ('female', 'female')],
        default='male'
    )

    can_write = models.BooleanField(default=False)
    can_speak_with_sign_language = models.BooleanField(default=False)

    #  assistant فقط لـ blind & deaf
    assistant = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patients',
        limit_choices_to={'user_type': 'assistant'}
    )

    is_active = models.BooleanField(default=True)

    REQUIRED_FIELDS = ['phone']

    # -----------------
    # Business rules
    # -----------------
    def clean(self):
        # assistant لا يكون له assistant
        if self.user_type == 'assistant' and self.assistant is not None:
            raise ValidationError("Assistant cannot have an assistant.")

        # blind / deaf لازم يكون لهم assistant
        if self.user_type in ['blind', 'deaf'] and self.assistant is None:
            raise ValidationError("Blind and Deaf users must have an assistant.")

        # volunteer لا يكون له assistant
        if self.user_type == 'volunteer' and self.assistant is not None:
            raise ValidationError("Volunteer cannot have an assistant.")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.username} - {self.user_type}"



class PasswordResetCode(models.Model):
    email = models.EmailField(db_index=True)
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    used = models.BooleanField(default=False)

    def is_expired(self) -> bool:
        return timezone.now() > self.created_at + timedelta(minutes=10)

    def __str__(self):
        return f"{self.email} ({self.code})"






from django.conf import settings
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    
    HELP_WAITING = 0
    IN_HELP_TRIP = 1
    COMPLETED = 2
    PINDING_APPROVAL= 3
    IN_PROGRESS=4
    CANCLED=5


    STATE_CHOICES = [
        (HELP_WAITING, 'Help Waiting'),
        (IN_HELP_TRIP, 'In Help Trip'),
        (COMPLETED, 'Completed'),
        (PINDING_APPROVAL, 'Pinding Approval'),
        (IN_PROGRESS, 'In Progress'),
        (CANCLED, 'Canceled')
    ]

    city = models.ForeignKey(
        "account.City",
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )

    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="posts"
    )
    volunteer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="helped_posts"
    )

    help_requesters = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name="requested_posts",
        blank=True
    )


    state = models.IntegerField(choices=STATE_CHOICES, default=HELP_WAITING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class City (models.Model):
    name = models.CharField(max_length=100, unique=True, default="there is no name")
    def __str__(self):
        return self.name