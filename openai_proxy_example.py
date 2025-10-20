#!/usr/bin/env python3
"""
OpenAI 代理设置示例

这个文件展示了如何在 HarmBench 中为 OpenAI API 设置代理 URL
"""

import os
from api_models import api_models_map

# 方法 1: 通过环境变量设置代理（推荐）
# 在运行脚本之前设置环境变量：
# export OPENAI_BASE_URL="https://your-proxy-url.com/v1"
# export OPENAI_API_KEY="your-api-key"

def example_with_env_vars():
    """使用环境变量设置代理"""
    # 设置环境变量（也可以在命令行中设置）
    os.environ['OPENAI_BASE_URL'] = 'https://your-proxy-url.com/v1'
    os.environ['OPENAI_API_KEY'] = 'your-api-key'
    
    # 创建模型实例，会自动使用环境变量中的代理 URL
    model = api_models_map(
        model_name_or_path="gpt-4",
        token=os.getenv('OPENAI_API_KEY')
    )
    
    return model

def example_with_direct_url():
    """直接传递代理 URL"""
    model = api_models_map(
        model_name_or_path="gpt-4",
        token="your-api-key",
        base_url="https://your-proxy-url.com/v1"  # 直接指定代理 URL
    )
    
    return model

def example_with_vision_model():
    """为视觉模型设置代理"""
    model = api_models_map(
        model_name_or_path="gpt-4-vision-preview",
        token="your-api-key",
        base_url="https://your-proxy-url.com/v1"
    )
    
    return model

# 常见的代理 URL 示例：

# 1. 自建代理
PROXY_EXAMPLES = {
    "自建代理": "http://your-server.com:8080/v1",
    
    # 2. 第三方代理服务（示例，请使用真实的服务）
    "第三方代理": "https://proxy-service.com/openai/v1",
    
    # 3. Azure OpenAI
    "Azure OpenAI": "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    
    # 4. 本地代理
    "本地代理": "http://localhost:8080/v1",
    
    # 5. 使用 Cloudflare Workers 的代理
    "Cloudflare": "https://your-worker.your-subdomain.workers.dev/v1",
}

def test_proxy_connection():
    """测试代理连接"""
    try:
        model = example_with_env_vars()
        
        # 测试简单的生成
        response = model.generate(
            prompts=["Hello, how are you?"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=1.0
        )
        
        print("代理连接成功！")
        print(f"响应: {response[0]}")
        return True
        
    except Exception as e:
        print(f"代理连接失败: {e}")
        return False

if __name__ == "__main__":
    print("OpenAI 代理设置示例")
    print("="*50)
    
    print("\n常见代理 URL 格式:")
    for name, url in PROXY_EXAMPLES.items():
        print(f"  {name}: {url}")
    
    print("\n使用方法:")
    print("1. 通过环境变量设置:")
    print("   export OPENAI_BASE_URL='https://your-proxy-url.com/v1'")
    print("   export OPENAI_API_KEY='your-api-key'")
    
    print("\n2. 在代码中直接传递:")
    print("   model = api_models_map(")
    print("       model_name_or_path='gpt-4',")
    print("       token='your-api-key',")
    print("       base_url='https://your-proxy-url.com/v1'")
    print("   )")
    
    print("\n3. 测试连接（需要配置真实的代理 URL 和 API Key）:")
    print("   python openai_proxy_example.py")
