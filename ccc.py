import uuid
import time
import random

def generate_sign_out_code(course_id):
    # 获取当前的时间戳
    timestamp = int(time.time())
    
    # 生成一个随机数
    random_number = random.randint(1000, 9999)
    
    # 创建一个唯一标识符（UUID）
    unique_id = str(uuid.uuid4()).replace("-", "").upper()[:8]
    
    # 拼接成签退码
    sign_out_code = f"{course_id}-{timestamp}-{random_number}-{unique_id}"
    
    return sign_out_code

# 示例
course_id = "e05f82f1-dd0b-4a86-8c49-f0323b07bc2a"  # 假设从课程详情页面获取到的课程ID
sign_out_code = generate_sign_out_code(course_id)
print("生成的签退码:", sign_out_code)
