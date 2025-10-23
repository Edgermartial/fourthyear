# myapp/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ForumConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("forum", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("forum", self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get("message", "")

        # 👇 Determine sender
        sender = "Admin" if self.scope["user"].is_staff else self.scope["user"].username or "Guest"

        await self.channel_layer.group_send(
            "forum",
            {
                "type": "chat_message",
                "message": message,
                "sender": sender,
            }
        )

    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            "message": event["message"],
            "sender": event["sender"]
        }))
