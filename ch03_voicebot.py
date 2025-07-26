import streamlit as st
from audiorecorder import audiorecorder
from openai import OpenAI
import os
from datetime import datetime
import numpy as np
from gtts import gTTS
import base64
import io
from pydub import AudioSegment

def STT(audio, client): 
    filename = "input.mp3"

    with open(filename, "wb") as wav_file:
        if isinstance(audio, AudioSegment):
            audio.export(wav_file, format="mp3")  
        else:
            wav_file.write(audio)

    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    os.remove(filename)

    return transcript.text

def ask_gpt(prompt, model, client):
    response = client.chat.completions.create(
        model=model,
        messages=prompt
    )

    return response.choices[0].message.content

def TTS(response):
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)

    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    os.remove(filename)

def main():
    st.set_page_config(
        page_title="음성 비서 프로그램",
        layout="wide"
    )

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}
        ]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = None

    st.header("음성 비서 프로그램")
    st.markdown("---")

    with st.expander("음성 비서 프로그램에 관하여", expanded=True):
        st.write("""
            - 음성 비서 프로그램의 UI는 streamlit을 활용했습니다.
            - STT(Speech-To-Text)는 OpenAI의 Whisper AI를 활용했습니다.
            - 답변은 OpenAI의 GPT 모델을 활용했습니다.
            - TTS(Text-To-Speech)는 구글의 Google Translate TTS를 활용했습니다.
        """)
        st.markdown("")

    with st.sidebar:
        api_key_input = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")
        st.markdown("---")

        model = st.radio(label="GPT 모델", options=["gpt-4", "gpt-3.5-turbo"])
        st.markdown("---")

        if st.button(label="초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = [
                {"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}
            ]
            st.session_state["check_audio"] = None
            st.experimental_rerun()

    if not api_key_input:
        st.warning("OpenAI API 키를 입력해주세요.")
        return

    client = OpenAI(api_key=api_key_input)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")

        if len(audio) > 0 and (st.session_state["check_audio"] is None or not np.array_equal(audio, st.session_state["check_audio"])):
            if isinstance(audio, AudioSegment):
                bytes_io = io.BytesIO()
                audio.export(bytes_io, format="wav")
                bytes_io.seek(0)
                st.audio(bytes_io.read(), format="audio/wav")
            else:
                st.audio(audio)

            question = STT(audio, client)
            now = datetime.now().strftime("%H:%M")

            st.session_state["chat"].append(("user", now, question))
            st.session_state["messages"].append({"role": "user", "content": question})
            st.session_state["check_audio"] = audio
    with col2:
        st.subheader("질문/답변")

        if st.session_state["chat"]:
            if len(st.session_state["chat"]) > 0 and (len(st.session_state["messages"]) == 1 or st.session_state["chat"][-1][0] == "user"):
                response = ask_gpt(st.session_state["messages"], model, client)
                st.session_state["messages"].append({"role": "system", "content": response})
                now = datetime.now().strftime("%H:%M")
                st.session_state["chat"].append(("bot", now, response))
                TTS(response)

            for sender, time_, message in st.session_state["chat"]:
                if sender == "user":
                    st.markdown(f'<div style="display:flex;align-items:center;">'
                                f'<div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div>'
                                f'<div style="font-size:0.8rem;color:gray;">{time_}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="display:flex;align-items:center;justify-content:flex-end;">'
                                f'<div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div>'
                                f'<div style="font-size:0.8rem;color:gray;">{time_}</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
