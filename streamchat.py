import os
from typing import Generator, TypedDict

import openai


class Message(TypedDict):
    role: str
    content: str


openai.api_key = os.environ["OPENAI_API_KEY"]


def truncate_message_history(message_history: list[Message], char_limit: int = 1000) -> list[Message]:
    """
    Truncate the message history so that the total number of characters 
    in all the messages don't cross `char_limit`. 
    Messages should be removed from the start of the list, and only entire messages can be removed.
    """
    new_list: list[Message] = []
    total_chars = 0
    for message in reversed(message_history):
        total_chars += len(message['content'])
        if total_chars > char_limit:
            break
        new_list.append(message)
    return list(reversed(new_list))


def stream_chat(prompt: str, prev_conversation: list[Message]) -> Generator[str, None, None]:
    """
    Generator function that yields message chunks.
    """
    prev_conversation = truncate_message_history(prev_conversation, 100)
    for message in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            *prev_conversation,
            {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        if message['choices'][0]['finish_reason'] != None:
            break
        yield message['choices'][0]['delta']['content']


def main():
    print("This is a simple GPT-powered chat interface.")
    print("Starting a chat with GPT. Type 'quit' to end the conversation.")
    message_history: list[Message] = []
    while True:
        user_prompt = input("\n\nYou: ")
        if user_prompt.lower() == "quit":
            print("Exiting the chat interface.")
            break
        else:
            bot_message = ""
            print("ChatGPT: ", end='', flush=True)
            for msg_chunk in stream_chat(user_prompt, message_history):
                print(msg_chunk, end='', flush=True)
                bot_message += msg_chunk
            message_history.append(Message(role="user", content=user_prompt))
            message_history.append(Message(role="assistant", content=bot_message))


if __name__ == "__main__":
    main()
