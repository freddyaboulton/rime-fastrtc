import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    WebRTCError,
    get_current_context,
    get_cloudflare_turn_credentials_async,
    get_stt_model,
    aggregate_bytes_to_16bit,
)
import os
import requests
from huggingface_hub import InferenceClient

load_dotenv()

stt_model = get_stt_model()

conversations: dict[str, list[dict[str, str]]] = {}


def stream_speech(text: str, speaker: str, rime_token: str):
    url = "https://users.rime.ai/v1/rime-tts"

    payload = {
        "speaker": speaker,
        "text": text,
        "modelId": "arcana",
        "repetition_penalty": 1.5,
        "temperature": 0.5,
        "top_p": 1,
        "samplingRate": 24000,
        "max_tokens": 1200,
    }
    headers = {
        "Accept": "audio/pcm",
        "Authorization": "Bearer " + rime_token,
        "Content-Type": "application/json",
    }

    with requests.post(url, headers=headers, json=payload, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                yield chunk


def response(
    audio: tuple[int, np.ndarray],
    hf_token: str | None,
    rime_token: str | None,
    speaker: str,
):
    if hf_token is None or hf_token == "":
        raise WebRTCError("HF Token is required")
    if rime_token is None or rime_token == "":
        raise WebRTCError("RIME Token is required")

    llm_client = InferenceClient(provider="auto", token=hf_token)

    context = get_current_context()
    if context.webrtc_id not in conversations:
        conversations[context.webrtc_id] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can have engaging conversations."
                    "Your responses must be very short and concise. No more than two sentences. "
                    "Reasoning: low"
                ),
            }
        ]

    messages = conversations[context.webrtc_id]

    transcription = stt_model.stt(audio)

    messages.append({"role": "user", "content": transcription})
    yield AdditionalOutputs(messages)

    output = llm_client.chat.completions.create(  # type: ignore
        model="openai/gpt-oss-20b",
        messages=messages,  # type: ignore
        max_tokens=1024,
        stream=True,
    )

    output_text = ""
    for chunk in output:
        if chunk and len(chunk.choices):
            output_text += chunk.choices[0].delta.content or ""

    messages.append({"role": "assistant", "content": output_text})
    conversations[context.webrtc_id] = messages
    try:
        for arr in aggregate_bytes_to_16bit(
            stream_speech(output_text, speaker, rime_token)
        ):
            yield (24_000, arr)
    except Exception as e:
        raise WebRTCError(f"Error occurred while streaming speech: {e}")
    yield AdditionalOutputs(messages)


chatbot = gr.Chatbot(label="Chatbot", type="messages")
hf_token = gr.Textbox(
    label="HF Token",
    value=os.getenv("HF_TOKEN"),
    type="password",
)
rime_token = gr.Textbox(
    label="RIME Token",
    value=os.getenv("RIME_API_KEY"),
    type="password",
    placeholder="Enter your RIME API key here",
)
speaker = gr.Dropdown(
    label="Speaker",
    choices=["Luna", "Pola", "Ursa", "Sirius", "Andromeda"],
    value="Luna",
    allow_custom_value=True,
)

stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response),  # type: ignore
    rtc_configuration=get_cloudflare_turn_credentials_async,
    additional_inputs=[hf_token, rime_token, speaker],
    additional_outputs=[chatbot],
    additional_outputs_handler=lambda old, new: new,
    ui_args={"title": "Rime Arcana Conversational Agent (Powered by FastRTC ⚡️)"},
    time_limit=90,
    concurrency_limit=5,
)

if __name__ == "__main__":
    stream.ui.launch()
