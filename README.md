# Rime FastRTC Conversational Agent

A real-time conversational agent powered by FastRTC and Gradio. It streams audio, transcribes speech, and generates concise AI responses using Hugging Face models and RIME TTS.

## Features

- Real-time audio streaming and transcription
- Short, concise AI responses
- Multiple speaker voices via RIME TTS
- Secure authentication with HF and RIME tokens
- Gradio-based web UI

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install .
   ```
3. Set environment variables in a `.env` file:
   ```
   HF_TOKEN=your_hf_token
   RIME_API_KEY=your_rime_api_key
   ```
4. Run the app:
   ```bash
   python app.py
   ```

## Usage

Open the Gradio UI in your browser, enter your tokens, select a speaker, and start a conversation.

## License

MIT
