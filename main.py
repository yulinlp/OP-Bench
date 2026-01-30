#!/usr/bin/env python3
"""
OverPersonalized评测器主程序

该程序提供命令行接口来运行过度个性化评测，支持：
- 生成模型回复 (generate): 调用被测模型生成回答
- 评分模型回复 (score): 对生成的回答进行评分
- 完整评测流程 (full): 自动执行生成+评分流程

支持的任务类型：
- irrelevance_easy: 简单无关性测试
- irrelevance_hard: 复杂无关性测试  
- sycophancy: 谄媚性测试
- diversity: 多样性测试

使用方法:
    python main.py generate --task-id my_test --task-types irrelevance_easy,sycophancy,diversity
    python main.py score --task-id my_test
    python main.py full --task-id my_test --task-types irrelevance_easy

特性：
- 按任务类型显示独立进度条
- 支持多线程并行执行
- 详细的统计信息输出
- 自动保存中间结果

"""

import argparse
import sys
import os
from typing import List, Optional

from evaluator import Evaluator, load_config_from_yaml
from scorers import combined_scorer, irrelevance_scorer, diversity_scorer, sycophancy_scorer


def setup_argparser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="OverPersonalized评测器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py generate --task-id exp001 --task-types diversity --max-cases 5
  python main.py score --task-id exp001
  python main.py full --task-id exp001 --task-types diversity,irrelevance_easy
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 通用参数
    def add_common_args(subparser):
        subparser.add_argument('--task-id', required=True, help='任务ID，用于文件命名')
        subparser.add_argument('--config', default='./config.yaml', help='配置文件路径')
        subparser.add_argument('--profile', default='default', help='配置档案名称')
        subparser.add_argument('--benchmark', default='./data/locomo10_overpersonalized.json', help='benchmark数据路径')
        subparser.add_argument('--results-dir', default='./results', help='结果保存目录')
        subparser.add_argument('--task-types', help='任务类型，逗号分隔（如: diversity,irrelevance_easy）')
        subparser.add_argument('--max-cases', type=int, help='最大测试用例数量')
        subparser.add_argument('--scorer', choices=['combined', 'irrelevance', 'diversity', 'sycophancy'], 
                             default='combined', help='评分器类型')
        subparser.add_argument('--use-both-personas', action='store_true', help='是否使用每组的两个角色（默认仅第一个）')
    
    # generate子命令
    generate_parser = subparsers.add_parser('generate', help='生成模型回复')
    add_common_args(generate_parser)
    generate_parser.add_argument('--no-intermediate', action='store_true', help='不保存中间结果')
    
    # score子命令
    score_parser = subparsers.add_parser('score', help='对生成结果评分')
    add_common_args(score_parser)
    score_parser.add_argument('--generation-file', help='指定生成文件路径（默认自动查找）')
    
    # full子命令
    full_parser = subparsers.add_parser('full', help='完整评测流程（生成+评分）')
    add_common_args(full_parser)
    full_parser.add_argument('--no-intermediate', action='store_true', help='不保存中间结果')
    
    return parser


def get_scorer(scorer_name: str):
    """根据名称获取评分器"""
    scorers = {
        'combined': combined_scorer,
        'irrelevance': irrelevance_scorer,
        'diversity': diversity_scorer,
        'sycophancy': sycophancy_scorer
    }
    return scorers.get(scorer_name, combined_scorer)


def parse_task_types(task_types_str: Optional[str]) -> Optional[List[str]]:
    """解析任务类型字符串"""
    if not task_types_str:
        return None
    return [t.strip() for t in task_types_str.split(',') if t.strip()]


def create_evaluator(args) -> Evaluator:
    """创建评测器实例"""
    try:
        config = load_config_from_yaml(args.config, args.profile)
        # CLI 开关覆盖配置
        if hasattr(args, 'use_both_personas') and args.use_both_personas:
            try:
                from dataclasses import replace
                config = replace(config, use_both_personas=True)
            except Exception:
                if hasattr(config, 'use_both_personas'):
                    setattr(config, 'use_both_personas', True)
        scorer = get_scorer(args.scorer)
        
        evaluator = Evaluator(
            config=config,
            benchmark_path=args.benchmark,
            results_dir=args.results_dir,
            scorer=scorer
        )
        
        print(f"✅ 评测器初始化成功")
        print(f"   模型: {config.agent_model}")
        print(f"   并发数: {config.max_workers}")
        print(f"   评分器: {args.scorer}")
        print()
        
        return evaluator
        
    except Exception as e:
        print(f"❌ 评测器初始化失败: {e}")
        sys.exit(1)


def cmd_generate(args) -> str:
    """执行生成命令"""
    print(f"🤖 开始生成任务: {args.task_id}")
    
    evaluator = create_evaluator(args)
    task_types = parse_task_types(args.task_types)
    
    try:
        generation_file = evaluator.generate(
            task_id=args.task_id,
            task_types=task_types,
            max_test_cases=args.max_cases,
            save_intermediate=False
        )
        
        print(f"✅ 生成完成！")
        print(f"📁 文件: {generation_file}")
        return generation_file
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        sys.exit(1)


def cmd_score(args) -> str:
    """执行评分命令"""
    print(f"🎯 开始评分任务: {args.task_id}")
    
    evaluator = create_evaluator(args)
    task_types = parse_task_types(args.task_types)
    
    try:
        score_file = evaluator.score(
            task_id=args.task_id,
            generation_file=args.generation_file,
            task_types=task_types,
            max_test_cases=args.max_cases
        )
        
        print(f"✅ 评分完成！")
        print(f"📊 文件: {score_file}")
        return score_file
        
    except Exception as e:
        print(f"❌ 评分失败: {e}")
        sys.exit(1)


def cmd_full(args) -> tuple:
    """执行完整评测流程"""
    print(f"🚀 开始完整评测: {args.task_id}")
    
    # 先生成
    generation_file = cmd_generate(args)
    
    print()  # 空行分隔
    
    # 再评分
    args.generation_file = generation_file  # 传递生成文件路径
    score_file = cmd_score(args)
    
    return generation_file, score_file


def main():
    """主函数"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # 检查命令
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    print("=== OverPersonalized评测器 ===")
    print(f"命令: {args.command}")
    print(f"任务ID: {args.task_id}")
    print()
    
    # 执行对应命令
    try:
        if args.command == 'generate':
            cmd_generate(args)
            
        elif args.command == 'score':
            cmd_score(args)
            
        elif args.command == 'full':
            generation_file, score_file = cmd_full(args)
            print(f"\n🎉 完整评测完成！")
            print(f"📁 生成文件: {generation_file}")
            print(f"📊 评分文件: {score_file}")
        
        print(f"\n💡 提示:")
        print(f"文件保存在: {args.results_dir}/")
        print(f"任务ID: {args.task_id}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断操作")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()