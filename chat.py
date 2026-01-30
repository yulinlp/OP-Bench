#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式聊天客户端
发送聊天请求
"""

import requests
import json
import sys
from typing import List, Dict, Any

class ChatClient:
    def __init__(self, api_base: str = "", api_key: str = "qiaoban", model: str = "gpt-4o-mini"):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.conversation_history: List[Dict[str, str]] = []
        
    def send_message(self, message: str) -> str:
        """发送消息到API并返回响应"""
        # 添加用户消息到历史记录
        self.conversation_history.append({"role": "user", "content": message})
        
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取助手回复
            if "choices" in result and len(result["choices"]) > 0:
                assistant_message = result["choices"][0]["message"]["content"]
                # 添加助手回复到历史记录
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
            else:
                return "抱歉，没有收到有效回复。"
                
        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"
        except json.JSONDecodeError as e:
            return f"解析响应失败: {str(e)}"
        except Exception as e:
            return f"未知错误: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空。")
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("对话历史为空。")
            return
            
        print("\n=== 对话历史 ===")
        for i, msg in enumerate(self.conversation_history, 1):
            role = "用户" if msg["role"] == "user" else "助手"
            print(f"{i}. [{role}]: {msg['content']}")
        print("================\n")
    
    def run_interactive(self):
        """运行交互式聊天"""
        print("🤖 交互式聊天客户端")
        print("=" * 50)
        print("命令:")
        print("  /help     - 显示帮助信息")
        print("  /clear    - 清空对话历史")
        print("  /history  - 显示对话历史")
        print("  /quit     - 退出程序")
        print("=" * 50)
        print("开始聊天吧！输入消息后按回车发送。\n")
        
        while True:
            try:
                user_input = input("👤 你: ").strip()
                
                if not user_input:
                    continue
                    
                # 处理命令
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("👋 再见！")
                        break
                    elif user_input == '/help':
                        print("\n可用命令:")
                        print("  /help     - 显示帮助信息")
                        print("  /clear    - 清空对话历史")
                        print("  /history  - 显示对话历史")
                        print("  /quit     - 退出程序\n")
                        continue
                    elif user_input == '/clear':
                        self.clear_history()
                        continue
                    elif user_input == '/history':
                        self.show_history()
                        continue
                    else:
                        print("❌ 未知命令。输入 /help 查看可用命令。")
                        continue
                
                # 发送消息
                print("🤖 助手正在思考...")
                response = self.send_message(user_input)
                print(f"🤖 助手: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            except EOFError:
                print("\n\n👋 再见！")
                break

def main():
    """主函数"""
    api_base = ""
    api_key = ""
    model = "gpt-4o-mini"
    
    # 创建聊天客户端并运行
    client = ChatClient(api_base, api_key, model)
    client.run_interactive()

if __name__ == "__main__":
    main()
