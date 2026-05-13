import asyncio
import base64
from io import BytesIO
from typing_extensions import override  # ✅ Fix 1: import override

from PIL import Image
import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskNotFoundError,    # ✅ Fix 2: import TaskNotFoundError
)
from a2a.utils.errors import ServerError  # ✅ Fix 2: import ServerError
from a2a.utils import new_agent_text_message

from specialist_agent.prompt import SPECIALIST_PROMPT


class MedGemmaAgentExecutor(AgentExecutor):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # ✅ Fix 6: correct way to get user input in A2A
        user_prompt = context.get_user_input() or "Analyze the morphology of the lesion in this image."

        # Get image from message parts if provided
        image_b64 = None
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part.root, 'data'):
                    image_b64 = part.root.data
                    break

        if not image_b64:
            event_queue.enqueue_event(  # ✅ Fix 5: enqueue_event not put_event
                new_agent_text_message("No image provided.")
            )
            return

        try:
            # Decode image
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Build input using the proper Gemma 3 / MedGemma chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": SPECIALIST_PROMPT + "\n\n" + user_prompt}
                    ]
                }
            ]

            loop = asyncio.get_event_loop()

            def run_inference():
                # apply_chat_template handles tokenisation + image encoding
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device, dtype=self.model.dtype)

                input_len = inputs["input_ids"].shape[-1]
                with __import__("torch").inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False
                    )
                # Decode only the newly generated tokens
                return self.processor.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                )

            result = await loop.run_in_executor(None, run_inference)

            # ✅ Fix 5: correct method to send result
            event_queue.enqueue_event(new_agent_text_message(result))

        except Exception as e:
            event_queue.enqueue_event(new_agent_text_message(f"Error: {str(e)}"))

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