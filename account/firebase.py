import datetime
import firebase_admin
from firebase_admin import credentials, messaging
from django.conf import settings

_firebase_app = None

def get_firebase_app():
    global _firebase_app
    if _firebase_app is None:
        cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
        _firebase_app = firebase_admin.initialize_app(cred)
    return _firebase_app


def send_fcm_to_token(
    token: str,
    title: str,
    body: str,
    data: dict | None = None,
    android_channel_id: str | None = None,
    is_call: bool = False,
    ttl_seconds: int | None = None,
):
    """
    Normal notification:
      - sends notification(title/body) + data

    Incoming call (is_call=True):
      - sends data-only (notification=None) + high priority + optional TTL
      - app should handle UI + ringtone
    """
    get_firebase_app()

    safe_data = {str(k): str(v) for k, v in (data or {}).items()}
    ttl = datetime.timedelta(seconds=int(ttl_seconds)) if ttl_seconds else None

    message = messaging.Message(
        token=token,

        # ✅ data-only for calls
        notification=None if is_call else messaging.Notification(title=title, body=body),

        data=safe_data,

        android=messaging.AndroidConfig(
            priority="high",
            ttl=ttl,
            notification=(
                None if is_call else messaging.AndroidNotification(
                    channel_id=android_channel_id,
                    sound="default",
                )
            ),
        ),

        apns=messaging.APNSConfig(
            headers={"apns-priority": "10"},
            payload=messaging.APNSPayload(
                aps=messaging.Aps(sound="default")
            )
        )
    )

    return messaging.send(message)
