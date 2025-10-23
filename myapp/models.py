from django.db import models
from django.contrib.auth.models import User

class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    is_admin_reply = models.BooleanField(default=False)  # mark admin replies
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        role = "Admin" if self.is_admin_reply else self.user.username if self.user else "Guest"
        return f"{role}: {self.content[:30]}"

class ChatMessage(models.Model):
    name = models.CharField(max_length=100, blank=True)
    email = models.EmailField(blank=True)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    reply = models.TextField(blank=True, null=True)  # Admin reply
    replied_at = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"Message from {self.name or 'Anonymous'}"

