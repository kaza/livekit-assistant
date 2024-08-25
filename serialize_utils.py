import json

from livekit.agents.llm import FunctionCallInfo, ChatContext, ChatImage, ChatMessage


def function_arg_info_to_dict(arg_info):
    return {
        "name": arg_info.name,
        "description": arg_info.description,
        "type": str(arg_info.type),  # Convert type to string for JSON serialization
        "default": str(arg_info.default),  # Convert default to string for JSON serialization
        "choices": arg_info.choices
    }


def function_info_to_dict(function_info):
    return {
        "name": function_info.name,
        "description": function_info.description,
        "auto_retry": function_info.auto_retry,
        "callable": str(function_info.callable),  # Convert callable to string for JSON serialization
        "arguments": {k: function_arg_info_to_dict(v) for k, v in function_info.arguments.items()},
        "raw_arguments": getattr(function_info, 'raw_arguments', None)  # Safely access raw_arguments
    }


def function_call_info_to_dict(function_call_info):
    return {
        "tool_call_id": function_call_info.tool_call_id,
        "function_info": function_info_to_dict(function_call_info.function_info)
    }


def debug_tool_calls(chat_message):

    # Print the type and content of tool_calls
    if chat_message.tool_calls is not None:
        print("tool_calls type:", type(chat_message.tool_calls))
        print("tool_calls content:", chat_message.tool_calls)
    else:
        print("tool_calls is None")


def chat_image_to_dict(chat_image):
    return {
        "inference_width": chat_image.inference_width,
        "inference_height": chat_image.inference_height
    }


def chat_message_to_dict(chat_message: ChatMessage):
    #debug_tool_calls(chat_message)

    # Handle content if it includes ChatImage objects
    content_serialized = []
    if isinstance(chat_message.content, list):
        for item in chat_message.content:
            if isinstance(item, ChatImage):
                content_serialized.append(chat_image_to_dict(item))
            else:
                content_serialized.append(item)
    else:
        content_serialized = chat_message.content

    # Convert the ChatMessage object to a dictionary
    return {
        "role": str(chat_message.role),  # Assuming ChatRole can be converted to a string
        "name": chat_message.name,
        "content": content_serialized,
        "tool_calls": [
            function_call_info_to_dict(call) if isinstance(call, FunctionCallInfo) else call
            for call in chat_message.tool_calls
        ] if chat_message.tool_calls else None,
        "tool_call_id": chat_message.tool_call_id
    }


def chat_context_to_dict(chat_context: ChatContext):
    return {
        "messages": [chat_message_to_dict(msg) for msg in chat_context.messages],
        "metadata": chat_context._metadata
    }


def store_context(name: str, chat_context: ChatContext):
    file_name = f"logs/{name}.json"
    try:
        # Convert the ChatContext object to a dictionary
        context_dict = chat_context_to_dict(chat_context)

        # Serialize the dictionary to a JSON file
        with open(file_name, 'w') as json_file:
            json.dump(context_dict, json_file, indent=4)

        print(f"Context successfully stored in {file_name}")
    except Exception as e:
        print(f"An error occurred while storing the context: {e}")
