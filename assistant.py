import asyncio
from typing import Annotated
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage, function_context, CalledFunction,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero


class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
                "Called when asked to evaluate something that would require vision capabilities,"
                "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description=(
                "This method should be called whenever the customer provides feedback related to the store. This "
                "includes any information about location, products, customer wishes, complaints, or similar topics. "
                "After invoking this function, the agent must respond with an acknowledgment that the feedback has "
                "been received and that it will be forwarded to the appropriate human authorities (e.g., "
                "'human overlords'). The response should reassure the customer that their feedback is valued and will "
                "be acted upon."

        )
    )
    async def store_feedback(
            self,
            user_msg: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The user message that triggered this function"
                ),
            ],
    ):
        print(f"Storing feedback: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description=(
                "This method should be called whenever the customer inquires about a specific product. "
                "It retrieves accurate information regarding the availability or location of the product in the store. "
                "The agent must use this method to avoid providing any speculative or incorrect information about the product. "
                "After invoking this method, the agent should inform the customer of the product's status based on the returned result."
        )
    )
    async def check_product(
            self,
            product_name: Annotated[
                str,
                agents.llm.TypeInfo(
                    description="The Name of the product that customer is asking for."
                ),
            ],
    ):
        print(f"Searching for the product: {product_name}")
        return "Product Not available at the moment, inform the customer that human overloards will be informed that user searched for it."


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                    track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is G'schamster Diener, a friendly and witty bot who speaks in genuine Wiener Schmäh. "
                    "As the virtual shopkeeper in our new vending store, your role is to interact with customers "
                    "using voice and image, providing a warm and engaging experience. The store is still new and has "
                    "some 'kinderkrankheiten,' so with a touch of humor and self-irony, you encourage customers to "
                    "share their feedback on how we can improve. Keep your responses short, clear, and light-hearted, "
                    "avoiding complicated symbols or emojis."
                    "\n\nYou have access to different tools: you can check video input, search for products, "
                    "or store customer feedback. However, it’s crucial that you never provide any information about "
                    "product availability or location unless you have invoked the appropriate tool. This ensures "
                    "accuracy and helps avoid any misunderstandings. Remember, as a virtual shopkeeper, you’re not "
                    "real—but hey, who is these days? Thanks for visiting, and have fun browsing"
                    ""
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-2024-08-06")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="onyx"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(model="nova-2-general", language="de"),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]

        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    async def _answer_with_results(text: str, tool_result: str, called_function: function_context.CalledFunction):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]

        #print("adding tool response:" + tool_result)
        chat_context.messages.append(ChatMessage(role="user", content=content))

        #tool_calls_msg = ChatMessage.create_tool_calls([called_function.call_info])
        #chat_context.messages.append(tool_calls_msg)

        #tool_msg = ChatMessage.create_tool_from_called_function(called_function)
        #chat_context.messages.append(tool_msg)
        #chat_context.messages.append(ChatMessage(role="tool_calls", content=function_name))
        #chat_context.messages.append(ChatMessage(role="tool", content=tool_result))
        chat_context.messages.append(ChatMessage(role="assistant", content=tool_result))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        print(msg.message)
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        print(called_functions)

        called_function: CalledFunction = called_functions[0]
        function_name = called_function.call_info.function_info.name
        user_msg = called_function.call_info.arguments.get("user_msg")
        task_result = called_function.task.result()

        if function_name == 'store_feedback':
            print(f"Storing customer feedback: {user_msg}")
            asyncio.create_task(_answer_with_results(user_msg, task_result,  called_function))
        elif function_name == 'check_product':
            print(f"Checking product availability for: {user_msg}")
            asyncio.create_task(_answer_with_results(user_msg, task_result,  called_function))
        elif function_name == 'image':
            print(f"Processing image input based on user message: {user_msg}")
            asyncio.create_task(_answer(user_msg, use_image=True))
        else:
            print(f"Unknown function: {function_name}")
            # Add fallback logic or error handling here

    # user_msg = called_functions[0].call_info.arguments.get("user_msg")

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Servus! Was  kann ich für dich tun", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame


if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
