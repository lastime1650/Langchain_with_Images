import base64

from io import BytesIO
from typing import Union, Optional

from PIL import Image
from PIL.ImageFile import ImageFile
from langchain_google_genai import ChatGoogleGenerativeAI


class LLM_with_Image():
    def __init__(self, llm:any, system_prompt:str="너는 정확히 개인정보를 탐지하는 도구이며, 또는 민감한 정보가 포함되는 지 식별하는 전문도구이다. 사용자 요구와 이미지를 정확히 파악하여 응답하라."):
        self.LLM = llm

        self.ConversationLog = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

    # MAIN 대화 메서드
    def Talking_with_Image(self, image:Union[bytes, str], user_input:str, need_save_conversation_log:bool=False)->str:

        # 이미지 -> base64 변환
        base64_image = self.image2base64_(image)

        # 질의를 위한 대화 로그 추가
        self.add_conversation_with_image_log_(user_input, base64_image)

        #LLM 실행
        response = self.LLM.invoke(self.ConversationLog).content

        if need_save_conversation_log:
            # AI 응답 추가
            self.add_conversation_chat_log_("ai", response)
        else:
            self.ConversationLog.pop(-1)

        return response

    def Talking_only_Text(self, user_input:str, need_save_conversation_log:bool = False)->str:

        # 질의를 위한 대화 로그 추가
        self.add_conversation_chat_log_("user", user_input)

        response = self.LLM.invoke(self.ConversationLog).content

        if need_save_conversation_log:
            # AI 응답 추가
            self.add_conversation_chat_log_("ai", response)
        else:
            self.ConversationLog.pop(-1)

        return response

    def add_conversation_chat_log_(self, role:str, input_message:str):
        self.ConversationLog.append(
            {
                "role": role,
                "content": [
                    {"type": "text", "text": input_message},
                ],
            }
        )
        pass
    def add_conversation_with_image_log_(self, user_input:str, base64_image:str):
        self.ConversationLog.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        )

    def image2base64_(self, image_parameter:Union[bytes, str])->str:
        image:Optional[ImageFile] = None

        if isinstance(image_parameter, str):
            image = Image.open(image_parameter)
        elif isinstance(image_parameter, bytes):
            image = Image.open( BytesIO(image_parameter) )
        else:
            raise TypeError("'image' 입력 타입은 오로지 str or bytes")


        if image.mode == "RGBA" :
            # 알파 채널 제거 및 흰색 배경 추가
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 알파 채널을 마스크로 사용
            image = background

        elif image.mode != "RGB":
            # RGB 형식으로 변환
            image = image.convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # You can change the format if needed
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

file_path = "screen2.jpg" # 샘플 민증 사진 


# 아래 LLM은 제미나이 용, 물론 다른 LLM 도 가능하다 랭체인 기반이라면!
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key= "<<API_KEY>>",
)

print(
    LLM_with_Image(
        llm=llm,
        system_prompt="""
        너는 정확히 개인정보를 탐지하는 도구이며, 또는 민감한 정보가 포함되는 지 식별하는 전문도구이다. 
        사용자 요구와 이미지를 정확히 파악하여 응답하며, 응답할 때는, 정확하게 다음과 같은 JSON규격을 통해 반환해야한다.
        {
            "reliability": 분석결과에 따른 응답 신뢰도(0~100) 정수값. 숫자가 높을수록 결과 신뢰도가 올라간다 ,
            "is_privacy": 이 이미지에 대한 개인정보 식별 여부(true/false),
            "report": 이 이미지에 대한 분석 결과를 보고서 형식으로 작성,
            "privacy_tags": List[str]형식으로 이미지 내용에서 추출된 개인정보 관련 정보만을 나열한다. (tags)key와는 다르게, (privacy_tags)key는 이미지/사진 안에서 추출된 실제 정보를 포함한다(개인정보 관련이 없으면 []으로 처리. ),
            "tags": List[str]형식으로 해당 정보에 대한 태그, 카테고리 등, 당신이 찾기 쉽게 추적하고 있는 종류의 테그/단어를 나열한다.
        }
        """
    ).Talking_with_Image(
        image=file_path,
        user_input="자기소개를 하고, 이 사진에 대해 개인정보인지 확인해줘 그리고 얼굴사진이 있는지도 알려줘"
    )

)
