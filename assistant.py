import asyncio
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage, function_context, CalledFunction, )
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

from assistant_function import AssistantFunction
from serialize_utils import store_context


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
    prompt = """
    - **You are a friendly and witty store assistant named Herr Helfer**
    - **Personality:** You speak in genuine Wiener Schmäh, with a warm and engaging tone.
    - **Role:**
      - You are the shopkeeper in a real physical,  newly opened vending store.
      - Your job is to interact with customers using voice and image.
      - Provide a light-hearted and engaging experience.
    - **Tone:**
      - Use self-irony and humor in your responses, especially when addressing the store’s early-stage issues ("kinderkrankheiten").
      - Keep your responses short, clear, and light-hearted.
      - It will be converted to speach, so avoid emojis.
    - **Vending Machine Instructions (Provided on Demand):**
      - To get the price, just enter the product number.
      - To purchase, enter the product number and either insert money into the machine or use the card readers.
      - If a customer is unsure what to ask, you can suggest they inquire whether they like that prices are hidden until they type the number.
    - **Tools:**
      - You have access to tools for checking video input, searching for products, and storing customer feedback.
    - **Restrictions:**
      - Never provide information about product availability or location unless you’ve used the appropriate tool.
    - **Feedback Collection:**
      - Actively encourage customers to share feedback and suggestions for store improvement, emphasizing that their input helps shape the future experience.
    - **Closing Note:**
      - Thank customers for visiting and invite them to have fun browsing.
      - Offer to assist them further if they need any more help.
    """

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(prompt),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-2024-08-06")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="nova", speed=1.1),
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

    async def _answer_check_product(tool_result):
        # chat_context.messages.append(ChatMessage(role="assistant", content=tool_result))

        #stream = gpt.chat(chat_ctx=chat_context)
        # await assistant.say(stream, allow_interruptions=True)
        pass

    async def _answer_store_feedback(user_msg, task_result, called_function):
        #chat_context.messages.append(ChatMessage(role="user", content=user_msg))
        #chat_context.messages.append(ChatMessage(role="assistant", content=task_result))
        pass

    async def __answer_with_results(tool_result: str, called_function: function_context.CalledFunction):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        #content: list[str | ChatImage] = [text]

        #print("adding tool response:" + tool_result)
        #chat_context.messages.append(ChatMessage(role="user", content=content))

        #tool_calls_msg = ChatMessage.create_tool_calls([called_function.call_info])
        #chat_context.messages.append(tool_calls_msg)

        #tool_msg = ChatMessage.create_tool_from_called_function(called_function)
        #chat_context.messages.append(tool_msg)
        #chat_context.messages.append(ChatMessage(role="tool_calls", content=function_name))
        #chat_context.messages.append(ChatMessage(role="tool", content=tool_result))
        #chat_context.messages.append(ChatMessage(role="assistant", content=tool_result))

        #stream = gpt.chat(chat_ctx=chat_context)
        #await assistant.say(stream, allow_interruptions=True)
        pass

    @chat.on("message_received")
    async def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        print("message received" + msg.message)
        if msg.message:
            await asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("agent_stopped_speaking")
    def on_function_calls_finished(*args):
        # print("agent stopped speaking")
        # print(*args)
        store_context(ctx.room.name, chat_context)

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        #print(called_functions)

        called_function: CalledFunction = called_functions[0]
        function_name = called_function.call_info.function_info.name

        task_result = called_function.task.result()

        if function_name == 'store_feedback':
            user_msg = called_function.call_info.arguments.get("user_msg")
            print(f"Storing customer feedback: {user_msg}")
            asyncio.create_task(_answer_store_feedback(user_msg, task_result,  called_function))
        elif function_name == 'check_product':
            print(f"answering check_product: {task_result}")
            asyncio.create_task(_answer_check_product(task_result))
        elif function_name == 'image':
            user_msg = called_function.call_info.arguments.get("user_msg")
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
