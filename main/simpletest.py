import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from.env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a new log file with timestamp in the main directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"main/chat_history_{timestamp}.txt"

# Prompt
prompt = "Be very professional Bank analyst"

# 초기 대화설정
messages = [
    {
        "role": "system",
        "content": prompt
    }
]

# Save conversation to file
def save_to_log(role, content):
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {role}: {content}\n")
        f.write("-" * 80 + "\n")

# Save system prompt
save_to_log("System", prompt)

# exit가 입력되기 전까지 계속 대화할거예요
while True:
    # 사용자 입력 받기
    user_input = input("User: ")
    # "exit" 입력 시 대화 종료
    if user_input.lower() == "exit":
        print("대화를 종료합니다")
        print(f"Conversation saved to: {log_file}")
        break
    
    # Save user input to log
    save_to_log("User", user_input)
    
    # 사용자 입력 값을 대화 목록에 추가 (대화의 흐름을 기억시키기 위해)
    messages.append({"role": "user", "content": user_input})
    
    # GPT3.5로 설정
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # GPT의 대답
    assistant_reply = response.choices[0].message.content
    # 대답 출력
    print(f"GPT: {assistant_reply}\n")
    
    # Save assistant reply to log
    save_to_log("Assistant", assistant_reply)

    # GPT 출력 값을 대화 목록에 추가 (대화의 흐름을 기억시키기 위해)
    messages.append({"role": "assistant", "content": assistant_reply})