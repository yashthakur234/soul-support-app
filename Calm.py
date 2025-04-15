import streamlit as st
import ollama
import base64
from textblob import TextBlob
import nltk
from datetime import datetime
import pandas as pd
import time
import speech_recognition as sr  # SpeechRecognition module for voice input

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")

# Background setup
def get_base64(background):
    try:
        with open(background, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error loading background image: {e}")
        return ""

bin_str = get_base64("background.png")

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .chat-box {{
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
        }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
st.session_state.setdefault('conversation_history', [])
st.session_state.setdefault('mood_history', [])
st.session_state.setdefault('current_mood', 'neutral')

# NLP Functions
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity < -0.5:
        return "😔 Stressed"
    elif polarity < 0:
        return "😟 Sad"
    elif polarity < 0.5:
        return "😊 Calm"
    else:
        return "😄 Happy"

# Mental Health Features
def track_mood(mood):
    st.session_state.mood_history.append({
        "mood": mood,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def breathing_exercise():
    return """🌬️ 4-7-8 Breathing Technique:
1. Empty your lungs completely
2. Breathe in quietly through nose for 4 seconds
3. Hold breath for 7 seconds
4. Exhale completely through mouth for 8 seconds
5. Repeat 4 times"""

def mental_health_resources():
    return """📚 Helpful Resources:
• Crisis Hotline: 1-800-273-TALK (8255)
• Anxiety & Depression Association of America: adaa.org
• National Suicide Prevention Lifeline: 988lifeline.org
• Calm Meditation App: calm.com
• Headspace Meditation: headspace.com"""

# Enhanced Response Generation
def generate_response(user_input):
    # Analyze sentiment
    mood = analyze_sentiment(user_input)
    track_mood(mood)
    st.session_state.current_mood = mood
    
    # Add context to prompt
    context = f"User mood detected as {mood}. Respond with empathy and emotional support."
    full_prompt = f"{context}\nUser: {user_input}"
    
    st.session_state.conversation_history.append({"role": "user", "content": full_prompt})
    
    response = ollama.chat(model="llama3:8b", messages=st.session_state.conversation_history)
    ai_response = response['message']['content']
    
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
    return ai_response

# Define the functions for affirmation and meditation guide
def generate_affirmation():
    prompt = f"Provide a positive affirmation for someone feeling {st.session_state.current_mood}"
    response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def generate_meditation_guide():
    prompt = f"Create a 5-minute guided meditation script for someone feeling {st.session_state.current_mood}"
    response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Function to listen to the microphone and convert speech to text
def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I did not understand that. Please try again.")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
    return None

# UI Components
st.title("Mental Health Support Agent 🌈")

# Sidebar for additional features
with st.sidebar:
    st.header("Additional Features")
    
    # Mood Tracking
    with st.expander("Track Your Mood"):
        mood = st.selectbox("How are you feeling?", ["😄 Happy", "😊 Calm", "😟 Sad", "😔 Stressed"])
        if st.button("Log Mood"):
            track_mood(mood)
            st.success("Mood logged successfully!")
    
    # Breathing Exercises
    if st.button("🧘 Breathing Exercise"):
        st.info(breathing_exercise())
    
    # Mental Health Resources
    if st.button("📚 Get Resources"):
        st.info(mental_health_resources())

# Main Chat Interface
with st.container():
    st.subheader("Chat Interface")
    
    # Chat history
    with st.container():
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for msg in st.session_state.conversation_history:
            role = "user" if msg['role'] == "user" else "assistant"
            with st.chat_message(role):
                st.write(msg['content'])
        st.markdown("</div>", unsafe_allow_html=True)
    
    # User input via text or microphone
    user_message = st.chat_input("How can I help you today?")
    
    if user_message:
        with st.spinner("Analyzing and responding..."):
            ai_response = generate_response(user_message)
            with st.chat_message("assistant"):
                st.write(ai_response)
                st.caption(f"Detected Mood: {st.session_state.current_mood}")
    
    # Listen via microphone button
    if st.button(""):
        speech_input = listen_to_microphone()
        if speech_input:
            ai_response = generate_response(speech_input)
            with st.chat_message("assistant"):
                st.write(ai_response)
                st.caption(f"Detected Mood: {st.session_state.current_mood}")
    
    # Clear Chat Button (added to the main chat interface)
    if st.button("🧹 Clear Chat"):
        st.session_state.conversation_history = []  # Clears the conversation history
        st.session_state.mood_history = []  # Clears the mood history
        st.rerun()  # Refreshes the app

# Mood Tracking Visualization
if st.session_state.mood_history:
    with st.expander("View Mood History"):
        mood_df = pd.DataFrame(st.session_state.mood_history)
        mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        
        # Count occurrences of each mood and plot as a bar chart
        mood_counts = mood_df['mood'].value_counts()
        st.bar_chart(mood_counts)

# Action Buttons
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌟 Get Affirmation"):
            affirmation = generate_affirmation()
            st.info(f"**Affirmation:** {affirmation}")
    
    with col2:
        if st.button("🧘 Guided Meditation"):
            meditation_guide = generate_meditation_guide()
            st.info(f"**Meditation Guide:** {meditation_guide}")
    
    with col3:
        if st.button("ℹ️ Current Mood"):
            st.info(f"Your current mood: {st.session_state.current_mood}")


# import streamlit as st
# import ollama
# import base64
# from textblob import TextBlob
# import nltk
# from datetime import datetime
# import pandas as pd
# import time
# import speech_recognition as sr  # SpeechRecognition module for voice input

# # Download NLTK data for sentiment analysis
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Set page config
# st.set_page_config(page_title="Mental Health Chatbot", layout="wide")

# # Background setup
# def get_base64(background):
#     try:
#         with open(background, "rb") as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
#     except Exception as e:
#         st.error(f"Error loading background image: {e}")
#         return ""

# bin_str = get_base64("background.png")

# st.markdown(f"""
#     <style>
#         .stApp {{
#             background-image: url("data:image/png;base64,{bin_str}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         .chat-box {{
#             max-height: 500px;
#             overflow-y: auto;
#             margin-bottom: 20px;
#             padding: 15px;
#             background: rgba(255, 255, 255, 0.9);
#             border-radius: 10px;
#         }}
#         /* Custom Microphone Button Styles */
#         .microphone-button {{
#             font-size: 24px;
#             background-color: transparent;
#             border: none;
#             cursor: pointer;
#         }}
#         .microphone-button img {{
#             width: 40px;
#             height: 40px;
#         }}
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# st.session_state.setdefault('conversation_history', [])
# st.session_state.setdefault('mood_history', [])
# st.session_state.setdefault('current_mood', 'neutral')

# # NLP Functions
# def analyze_sentiment(text):
#     analysis = TextBlob(text)
#     polarity = analysis.sentiment.polarity
#     if polarity < -0.5:
#         return "😔 Stressed"
#     elif polarity < 0:
#         return "😟 Sad"
#     elif polarity < 0.5:
#         return "😊 Calm"
#     else:
#         return "😄 Happy"

# # Mental Health Features
# def track_mood(mood):
#     st.session_state.mood_history.append({
#         "mood": mood,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     })

# def breathing_exercise():
#     return """🌬️ 4-7-8 Breathing Technique:
# 1. Empty your lungs completely
# 2. Breathe in quietly through nose for 4 seconds
# 3. Hold breath for 7 seconds
# 4. Exhale completely through mouth for 8 seconds
# 5. Repeat 4 times"""

# def mental_health_resources():
#     return """📚 Helpful Resources:
# • Crisis Hotline: 1-800-273-TALK (8255)
# • Anxiety & Depression Association of America: adaa.org
# • National Suicide Prevention Lifeline: 988lifeline.org
# • Calm Meditation App: calm.com
# • Headspace Meditation: headspace.com"""

# # Enhanced Response Generation
# def generate_response(user_input):
#     # Analyze sentiment
#     mood = analyze_sentiment(user_input)
#     track_mood(mood)
#     st.session_state.current_mood = mood
    
#     # Add context to prompt
#     context = f"User mood detected as {mood}. Respond with empathy and emotional support."
#     full_prompt = f"{context}\nUser: {user_input}"
    
#     st.session_state.conversation_history.append({"role": "user", "content": full_prompt})
    
#     response = ollama.chat(model="llama3:8b", messages=st.session_state.conversation_history)
#     ai_response = response['message']['content']
    
#     st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
#     return ai_response

# # Define the functions for affirmation and meditation guide
# def generate_affirmation():
#     prompt = f"Provide a positive affirmation for someone feeling {st.session_state.current_mood}"
#     response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
#     return response['message']['content']

# def generate_meditation_guide():
#     prompt = f"Create a 5-minute guided meditation script for someone feeling {st.session_state.current_mood}"
#     response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
#     return response['message']['content']

# # Function to listen to the microphone and convert speech to text
# def listen_to_microphone():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Please speak now.")
#         audio = recognizer.listen(source)
#         try:
#             text = recognizer.recognize_google(audio)
#             st.success(f"You said: {text}")
#             return text
#         except sr.UnknownValueError:
#             st.error("Sorry, I did not understand that. Please try again.")
#         except sr.RequestError:
#             st.error("Could not request results from Google Speech Recognition service.")
#     return None

# # UI Components
# st.title("Mental Health Support Agent 🌈")

# # Sidebar for additional features
# with st.sidebar:
#     st.header("Additional Features")
    
#     # Mood Tracking
#     with st.expander("Track Your Mood"):
#         mood = st.selectbox("How are you feeling?", ["😄 Happy", "😊 Calm", "😟 Sad", "😔 Stressed"])
#         if st.button("Log Mood"):
#             track_mood(mood)
#             st.success("Mood logged successfully!")
    
#     # Breathing Exercises
#     if st.button("🧘 Breathing Exercise"):
#         st.info(breathing_exercise())
    
#     # Mental Health Resources
#     if st.button("📚 Get Resources"):
#         st.info(mental_health_resources())

# # Main Chat Interface
# with st.container():
#     st.subheader("Chat Interface")
    
#     # Chat history
#     with st.container():
#         st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
#         for msg in st.session_state.conversation_history:
#             role = "user" if msg['role'] == "user" else "assistant"
#             with st.chat_message(role):
#                 st.write(msg['content'])
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # User input via text or microphone
#     user_message = st.chat_input("How can I help you today?")
    
#     if user_message:
#         with st.spinner("Analyzing and responding..."):
#             ai_response = generate_response(user_message)
#             with st.chat_message("assistant"):
#                 st.write(ai_response)
#                 st.caption(f"Detected Mood: {st.session_state.current_mood}")
    
#     # Custom Microphone Icon Button
#     mic_icon_url = "/Users/yashpratapsingh/Downloads/voice.svg"  # Replace this URL with the URL of your custom mic icon image
#     mic_button_html = f"""
#     <button class="microphone-button" onclick="window.streamlitApi.triggerStreamlitEvent('microphone_click')">
#         <img src="{mic_icon_url}" alt="Microphone Icon">
#     </button>
#     """
    
#     st.markdown(mic_button_html, unsafe_allow_html=True)

#     # Get the query parameters to detect if the microphone button was clicked
#     if 'microphone_click' in st.query_params:
#         speech_input = listen_to_microphone()
#         if speech_input:
#             ai_response = generate_response(speech_input)
#             with st.chat_message("assistant"):
#                 st.write(ai_response)
#                 st.caption(f"Detected Mood: {st.session_state.current_mood}")

#     # Clear Chat Button (added to the main chat interface)
#     if st.button("🧹 Clear Chat"):
#         st.session_state.conversation_history = []  # Clears the conversation history
#         st.session_state.mood_history = []  # Clears the mood history
#         st.experimental_rerun()  # Refreshes the app

# # Mood Tracking Visualization
# if st.session_state.mood_history:
#     with st.expander("View Mood History"):
#         mood_df = pd.DataFrame(st.session_state.mood_history)
#         mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        
#         # Count occurrences of each mood and plot as a bar chart
#         mood_counts = mood_df['mood'].value_counts()
#         st.bar_chart(mood_counts)

# # Action Buttons
# with st.container():
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("🌟 Get Affirmation"):
#             affirmation = generate_affirmation()
#             st.info(f"**Affirmation:** {affirmation}")
    
#     with col2:
#         if st.button("🧘 Guided Meditation"):
#             meditation_guide = generate_meditation_guide()
#             st.info(f"**Meditation Guide:** {meditation_guide}")
    
#     with col3:
#         if st.button("ℹ️ Current Mood"):
#             st.info(f"Your current mood: {st.session_state.current_mood}")


# import streamlit as st
# import ollama
# import base64
# from textblob import TextBlob
# import nltk
# from datetime import datetime
# import pandas as pd
# import time
# import speech_recognition as sr

# # ... [Keep all the initial imports and setup code the same] ...

# # Enhanced CSS Styling
# st.markdown(f"""
#     <style>
#         .stApp {{
#             background-image: url("data:image/png;base64,{bin_str}");
#             background-size: cover;
#             background-position: center;
#         }}
        
#         /* Professional Chat Container */
#         .chat-container {{
#             max-width: 800px;
#             margin: 0 auto;
#             background: rgba(255, 255, 255, 0.95);
#             border-radius: 15px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             padding: 2rem;
#         }}
        
#         /* User Message Styling */
#         .user-message {{
#             background: #e3f2fd;
#             border-radius: 15px 15px 0 15px;
#             padding: 1rem;
#             margin: 0.5rem 0;
#             max-width: 70%;
#             margin-left: auto;
#         }}
        
#         /* Bot Message Styling */
#         .bot-message {{
#             background: #f5f5f5;
#             border-radius: 15px 15px 15px 0;
#             padding: 1rem;
#             margin: 0.5rem 0;
#             max-width: 70%;
#         }}
        
#         /* Enhanced Input Area */
#         .input-container {{
#             position: fixed;
#             bottom: 2rem;
#             left: 50%;
#             transform: translateX(-50%);
#             width: 80%;
#             background: white;
#             padding: 1rem;
#             border-radius: 25px;
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#         }}
        
#         /* Professional Button Styling */
#         .stButton>button {{
#             border-radius: 20px;
#             padding: 0.5rem 1.5rem;
#             transition: all 0.3s ease;
#         }}
        
#         .stButton>button:hover {{
#             transform: translateY(-2px);
#             box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
#         }}
        
#         /* Mood Chart Container */
#         .mood-chart {{
#             background: white;
#             padding: 1.5rem;
#             border-radius: 15px;
#             margin-top: 2rem;
#         }}
#     </style>
# """, unsafe_allow_html=True)

# # ... [Keep the existing session state and function definitions] ...

# # Enhanced UI Components
# st.title("Mental Health Support Companion 🌱")

# # Professional Sidebar
# with st.sidebar:
#     st.markdown("""
#         <div style='border-bottom: 2px solid #e0e0e0; padding-bottom: 1rem; margin-bottom: 2rem;'>
#             <h2 style='color: #2c3e50;'>Wellness Toolkit 🧰</h2>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Mood Tracking with Emoji Picker
#     with st.expander("📅 Mood Tracker", expanded=True):
#         mood_options = {
#             "😄 Happy": "#4CAF50",
#             "😊 Calm": "#2196F3",
#             "😟 Sad": "#FFC107",
#             "😔 Stressed": "#F44336"
#         }
#         cols = st.columns(4)
#         for i, (mood, color) in enumerate(mood_options.items()):
#             with cols[i]:
#                 if st.button(mood, key=f"mood_{i}"):
#                     track_mood(mood)
#                     st.success(f"Logged: {mood}")
    
#     # Interactive Breathing Exercise
#     with st.expander("🌬️ Breathing Exercise"):
#         st.markdown("""
#             <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
#                 <h4>4-7-8 Technique</h4>
#                 <p style='color: #666;'>Follow the animated guide:</p>
#                 <div style='display: flex; justify-content: center;'>
#                     <div style='animation: breath 8s infinite; width: 50px; height: 50px; background: #2196F3; border-radius: 50%;'></div>
#                 </div>
#             </div>
#             <style>
#                 @keyframes breath {{
#                     0% {{ transform: scale(0.8); }}
#                     25% {{ transform: scale(1.2); }}
#                     50% {{ transform: scale(0.8); }}
#                     75% {{ transform: scale(1.2); }}
#                     100% {{ transform: scale(0.8); }}
#                 }}
#             </style>
#         """, unsafe_allow_html=True)
    
#     # Enhanced Resources Section
#     with st.expander("📚 Resources"):
#         st.markdown("""
#             <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
#                 <h4>Immediate Help</h4>
#                 <ul style='list-style: none; padding-left: 0;'>
#                     <li>🆘 Crisis Hotline: 1-800-273-TALK</li>
#                     <li>🌐 988lifeline.org</li>
#                 </ul>
#                 <h4>Self-Care Tools</h4>
#                 <ul style='list-style: none; padding-left: 0;'>
#                     <li>🧘 Calm App</li>
#                     <li>📱 Headspace</li>
#                 </ul>
#             </div>
#         """, unsafe_allow_html=True)

# # Main Chat Interface
# with st.container():
#     st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
#     # Chat History with Enhanced Styling
#     for msg in st.session_state.conversation_history:
#         if msg['role'] == "user":
#             st.markdown(f"""
#                 <div class='user-message'>
#                     <div style='color: #1a237e; font-weight: 500;'>You</div>
#                     <div style='margin-top: 0.5rem;'>{msg['content']}</div>
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#                 <div class='bot-message'>
#                     <div style='color: #0d47a1; font-weight: 500;'>Companion</div>
#                     <div style='margin-top: 0.5rem;'>{msg['content']}</div>
#                     <div style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>
#                         Mood Detected: {st.session_state.current_mood}
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Enhanced Input Area
# with st.container():
#     st.markdown("<div class='input-container'>", unsafe_allow_html=True)
#     col1, col2 = st.columns([6, 1])
    
#     with col1:
#         user_input = st.text_input("Share your thoughts...", label_visibility="collapsed")
    
#     with col2:
#         st.markdown("""
#             <style>
#                 .mic-button {{
#                     background: #2196F3;
#                     color: white;
#                     border: none;
#                     border-radius: 50%;
#                     width: 40px;
#                     height: 40px;
#                     display: flex;
#                     align-items: center;
#                     justify-content: center;
#                     transition: all 0.3s ease;
#                 }}
#                 .mic-button:hover {{
#                     background: #1976D2;
#                     transform: scale(1.1);
#                 }}
#             </style>
#             <button class='mic-button' onclick="window.streamlitApi.triggerStreamlitEvent('microphone_click')">
#                 🎤
#             </button>
#         """, unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # ... [Keep the remaining functionality the same] ...

# # Enhanced Mood Visualization
# if st.session_state.mood_history:
#     with st.expander("📈 Mood Analysis", expanded=True):
#         st.markdown("<div class='mood-chart'>", unsafe_allow_html=True)
#         mood_df = pd.DataFrame(st.session_state.mood_history)
#         mood_df['timestamp'] = pd.to_datetime(mood_df['timestamp'])
        
#         # Create time series chart
#         mood_df['time'] = mood_df['timestamp'].dt.strftime('%H:%M')
#         mood_chart = alt.Chart(mood_df).mark_line(point=True).encode(
#             x='time:T',
#             y=alt.Y('mood:N', sort=['😔 Stressed', '😟 Sad', '😊 Calm', '😄 Happy']),
#             color=alt.Color('mood:N', scale=alt.Scale(
#                 domain=['😔 Stressed', '😟 Sad', '😊 Calm', '😄 Happy'],
#                 range=['#F44336', '#FFC107', '#2196F3', '#4CAF50']
#             ))
#         ).properties(height=300)
        
#         st.altair_chart(mood_chart, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

