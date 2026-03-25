"""
模型转换工具
将DataParallel包装的多GPU模型转换为单GPU/CPU可用的模型
"""
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def convert_dataparallel_model(input_path, output_path):
    """
    转换DataParallel模型为普通模型
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
    """
    logging.info(f"加载模型: {input_path}")
    
    # 加载模型到CPU
    model = torch.load(input_path, map_location='cpu')
    
    # 检查是否是DataParallel模型
    if isinstance(model, torch.nn.DataParallel):
        logging.info("检测到DataParallel模型，正在提取原始模型...")
        model = model.module
        logging.info("✓ 成功提取原始模型")
    else:
        logging.info("模型不是DataParallel包装的，无需转换")
    
    # 保存转换后的模型
    logging.info(f"保存模型到: {output_path}")
    torch.save(model, output_path)
    logging.info("✓ 模型保存成功")
    
    # 验证转换后的模型
    logging.info("验证转换后的模型...")
    test_model = torch.load(output_path, map_location='cpu')
    if isinstance(test_model, torch.nn.DataParallel):
        logging.warning("⚠ 警告: 转换后的模型仍然是DataParallel")
    else:
        logging.info("✓ 验证成功，模型已正确转换")


def main():
    """主函数"""
    # 配置
    INPUT_MODEL = 'my_model_woXLSTM.pkl'  # 输入模型路径
    OUTPUT_MODEL = 'my_model_woXLSTM_converted.pkl'  # 输出模型路径
    
    try:
        convert_dataparallel_model(INPUT_MODEL, OUTPUT_MODEL)
        logging.info("\n转换完成！")
        logging.info(f"原始模型: {INPUT_MODEL}")
        logging.info(f"转换后模型: {OUTPUT_MODEL}")
        logging.info("\n现在可以使用转换后的模型进行测试了")
    except Exception as e:
        logging.error(f"转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
