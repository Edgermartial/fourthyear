from django.contrib import admin
from .models import Message

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("user", "content", "is_admin_reply", "created_at")
    list_filter = ("is_admin_reply", "created_at")
    search_fields = ("content", "user__username")


from django.contrib import admin
from django.core.mail import EmailMultiAlternatives
from django.utils.timezone import now
from django.utils.html import strip_tags
from django.template.loader import render_to_string
from .models import ChatMessage

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'short_message', 'created_at', 'is_read', 'has_reply')
    readonly_fields = ('name', 'email', 'message', 'created_at')
    fields = ('name', 'email', 'message', 'created_at', 'is_read', 'reply')

    # ✅ Define helper methods so Django admin knows what they are
    def short_message(self, obj):
        """Show a short preview of the message in admin list view"""
        return (obj.message[:40] + "...") if len(obj.message) > 40 else obj.message
    short_message.short_description = "Message Preview"

    def has_reply(self, obj):
        """Show True/False if admin has replied"""
        return bool(obj.reply)
    has_reply.boolean = True
    has_reply.short_description = "Replied?"

    def save_model(self, request, obj, form, change):
        """Sends email reply when admin adds a reply"""
        if obj.reply and not obj.replied_at:
            obj.replied_at = now()
            obj.is_read = True

            if obj.email:
                subject = "🌱 Reply from Climate Smart Agric Team"
                context = {
                    "name": obj.name or "Friend",
                    "reply": obj.reply,
                    "original_message": obj.message,
                }

                # Render HTML + plain text
                html_content = render_to_string("emails/reply_message.html", context)
                text_content = strip_tags(html_content)

                email = EmailMultiAlternatives(
                    subject,
                    text_content,
                    "climateapp02@gmail.com",
                    [obj.email],
                )
                email.attach_alternative(html_content, "text/html")
                email.send(fail_silently=True)

        super().save_model(request, obj, form, change)
