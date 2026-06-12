

from llamafactory.train.tuner import run_exp  # use absolute import

# import os
# from dotenv import load_dotenv
# from huggingface_hub import login

# # 从 .env 文件加载环境变量
# load_dotenv()

# # 从环境中获取 Token
# huggingface_token = os.getenv("HF_TOKEN")

# if huggingface_token:
#     print("成功从 .env 文件加载 Token!")
#     # 使用 Token 登录
#     login(token=huggingface_token)
#     print("Hugging Face Hub 登录成功！")

#     # 现在你可以执行需要身份验证的操作了
#     # 例如: from datasets import load_dataset
#     # dataset = load_dataset("meta-llama/Llama-2-7b-chat-hf")

# else:
#     print("错误：未能在 .env 文件中找到 HUGGING_FACE_HUB_TOKEN。")
    
    
    
def launch():
    run_exp()


if __name__ == "__main__":
    launch()
