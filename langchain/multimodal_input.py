from dotenv import load_dotenv
from base64 import b64encode

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model("gpt-4.1-mini")

message = HumanMessage(
    content = [
        {"type": "text", "text": "Descibe the contents of this image"},
            {
                "type": "image",
                "base64": b64encode(open("poison-dart-frog.jpg", "rb").read()).decode(),
                "mime_type": "image/png"
            }
        # {"type": "image", "url": "https://pngimg.com/uploads/rabbit/rabbit_PNG96521.png"}
    ]
)

# message = {
#     "role": "user",
#     "content": [
#         {"type": "text", "text": "Descibe the contents of this image"},
#             {
#                 "type": "image",
#                 "base64": b64encode(open("poison-dart-frog.jpg", "rb").read()).decode(),
#                 "mime_type": "image/png"
#             }
#         # {"type": "image", "url": "https://pngimg.com/uploads/rabbit/rabbit_PNG96521.png"}
#     ]
# }

response = model.invoke([message])

print(response.content)