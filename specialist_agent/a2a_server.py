import asyncio
import base64
from io import BytesIO
from typing_extensions import override

from PIL import Image
import uvicorn
from transformers import pipeline

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskNotFoundError,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message

from specialist_agent.prompt import SPECIALIST_PROMPT


class MedGemmaAgentExecutor(AgentExecutor):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        # Mirror the working reference notebook: drive inference through the
        # image-text-to-text pipeline rather than calling model.generate manually.
        self.pipe = pipeline(
            task="image-text-to-text",
            model=model,
            processor=processor,
        )
        print(f"[MedGemma] pipeline ready on device={self.pipe.device} dtype={model.dtype}")

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_prompt = context.get_user_input() or "Analyze the morphology of the lesion in this image."

        # Get image from message parts (A2A 0.3.x FilePart with inline bytes)
        image_b64 = None
        if context.message and context.message.parts:
            for part in context.message.parts:
                root = part.root
                if hasattr(root, "file") and hasattr(root.file, "bytes"):
                    image_b64 = root.file.bytes
                    break

        if not image_b64:
            await event_queue.enqueue_event(
                new_agent_text_message("No image provided.")
            )
            return

        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{SPECIALIST_PROMPT}\n\n{user_prompt}"},
                    ],
                }
            ]

            loop = asyncio.get_event_loop()

            def run_inference():
                output = self.pipe(text=messages, max_new_tokens=512)
                return output[0]["generated_text"][-1]["content"]

            result = await loop.run_in_executor(None, run_inference)
            await event_queue.enqueue_event(new_agent_text_message(result))

        except Exception as e:
            await event_queue.enqueue_event(new_agent_text_message(f"Error: {str(e)}"))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=TaskNotFoundError())


def create_a2a_app(model, processor):
    # ✅ Fix 3: AgentCard must be an object, not a dict
    agent_card = AgentCard(
        name="Dermatology Specialist",
        description="Analyzes skin lesion images using MedGemma VLM.",
        url="http://localhost:8001/",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
        ),
        skills=[
            AgentSkill(
                id="analyze_skin_lesion",
                name="Skin Lesion Analysis",
                description="Analyzes skin lesion images and provides detailed morphological findings",
                tags=["dermatology", "skin", "lesion"],
            )
        ],
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    task_store = InMemoryTaskStore()
    executor = MedGemmaAgentExecutor(model=model, processor=processor)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # ✅ Fix 4: correct parameter is http_handler, not request_handler
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    return app.build()  # ✅ return the built Starlette app


def run_server(app, port=8001):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()