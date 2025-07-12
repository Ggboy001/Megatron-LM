# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fp8_utils import is_float8tensor, is_mxfp8tensor, dequantize_fp8_tensor


class TestFP8TensorChecks:
    """测试FP8张量类型检查函数"""

    def test_is_float8tensor_regular_tensor(self):
        """测试普通张量不应该被识别为FP8张量"""
        # 创建普通张量
        regular_tensor = torch.randn(2, 3, dtype=torch.float32)
        assert not is_float8tensor(regular_tensor), "普通张量不应该被识别为FP8张量"
        
        # 测试不同数据类型的张量
        bf16_tensor = torch.randn(2, 3, dtype=torch.bfloat16)
        assert not is_float8tensor(bf16_tensor), "BF16张量不应该被识别为FP8张量"
        
        fp16_tensor = torch.randn(2, 3, dtype=torch.float16)
        assert not is_float8tensor(fp16_tensor), "FP16张量不应该被识别为FP8张量"

    def test_is_float8tensor_none_input(self):
        """测试None输入的处理"""
        # 测试None输入
        assert not is_float8tensor(None), "None输入应该返回False"

    def test_is_float8tensor_empty_tensor(self):
        """测试空张量的处理"""
        # 测试空张量
        empty_tensor = torch.empty(0, dtype=torch.float32)
        assert not is_float8tensor(empty_tensor), "空张量不应该被识别为FP8张量"

    def test_is_mxfp8tensor_regular_tensor(self):
        """测试普通张量不应该被识别为MXFP8张量"""
        # 创建普通张量
        regular_tensor = torch.randn(2, 3, dtype=torch.float32)
        assert not is_mxfp8tensor(regular_tensor), "普通张量不应该被识别为MXFP8张量"
        
        # 测试不同数据类型的张量
        bf16_tensor = torch.randn(2, 3, dtype=torch.bfloat16)
        assert not is_mxfp8tensor(bf16_tensor), "BF16张量不应该被识别为MXFP8张量"

    def test_is_mxfp8tensor_none_input(self):
        """测试None输入的处理"""
        # 测试None输入
        assert not is_mxfp8tensor(None), "None输入应该返回False"

    def test_is_mxfp8tensor_empty_tensor(self):
        """测试空张量的处理"""
        # 测试空张量
        empty_tensor = torch.empty(0, dtype=torch.float32)
        assert not is_mxfp8tensor(empty_tensor), "空张量不应该被识别为MXFP8张量"

    def test_tensor_type_consistency(self):
        """测试张量类型检查的一致性"""
        # 创建相同形状但不同数据类型的张量
        tensors = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float16),
            torch.randn(3, 4, dtype=torch.bfloat16),
            torch.randn(3, 4, dtype=torch.int32),
            torch.randn(3, 4, dtype=torch.int64),
        ]
        
        for tensor in tensors:
            # 对于普通张量，两个函数都应该返回False
            assert not is_float8tensor(tensor), f"{tensor.dtype}张量不应该被识别为FP8张量"
            assert not is_mxfp8tensor(tensor), f"{tensor.dtype}张量不应该被识别为MXFP8张量"

    def test_tensor_properties_preserved(self):
        """测试张量属性在检查过程中得到保持"""
        # 创建测试张量
        original_tensor = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        original_shape = original_tensor.shape
        original_dtype = original_tensor.dtype
        original_requires_grad = original_tensor.requires_grad
        
        # 进行类型检查
        is_float8 = is_float8tensor(original_tensor)
        is_mxfp8 = is_mxfp8tensor(original_tensor)
        
        # 验证张量属性没有改变
        assert original_tensor.shape == original_shape, "张量形状不应该改变"
        assert original_tensor.dtype == original_dtype, "张量数据类型不应该改变"
        assert original_tensor.requires_grad == original_requires_grad, "张量requires_grad属性不应该改变"

    def test_dequantize_fp8_tensor_regular_input(self):
        """测试反量化函数对普通张量的处理"""
        # 对于普通张量，反量化函数应该抛出异常或返回原张量
        regular_tensor = torch.randn(2, 3, dtype=torch.float32)
        
        try:
            result = dequantize_fp8_tensor(regular_tensor)
            # 如果没有抛出异常，结果应该是原张量
            assert torch.equal(result, regular_tensor), "普通张量反量化应该返回原张量"
        except Exception as e:
            # 如果抛出异常，这是可以接受的
            assert "fp8" in str(e).lower() or "float8" in str(e).lower(), f"异常信息应该与FP8相关: {e}"

    def test_dequantize_fp8_tensor_none_input(self):
        """测试反量化函数对None输入的处理"""
        try:
            result = dequantize_fp8_tensor(None)
            assert result is None, "None输入反量化应该返回None"
        except Exception as e:
            # 如果抛出异常，这是可以接受的
            assert "tensor" in str(e).lower() or "none" in str(e).lower(), f"异常信息应该与张量相关: {e}"

    def test_function_signatures(self):
        """测试函数签名和文档字符串"""
        # 检查函数是否存在
        assert callable(is_float8tensor), "is_float8tensor应该是可调用的"
        assert callable(is_mxfp8tensor), "is_mxfp8tensor应该是可调用的"
        assert callable(dequantize_fp8_tensor), "dequantize_fp8_tensor应该是可调用的"
        
        # 检查函数文档字符串
        assert is_float8tensor.__doc__ is not None, "is_float8tensor应该有文档字符串"
        assert is_mxfp8tensor.__doc__ is not None, "is_mxfp8tensor应该有文档字符串"
        assert dequantize_fp8_tensor.__doc__ is not None, "dequantize_fp8_tensor应该有文档字符串"

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试零维张量
        scalar_tensor = torch.tensor(1.0, dtype=torch.float32)
        assert not is_float8tensor(scalar_tensor), "零维张量不应该被识别为FP8张量"
        assert not is_mxfp8tensor(scalar_tensor), "零维张量不应该被识别为MXFP8张量"
        
        # 测试大张量
        large_tensor = torch.randn(1000, 1000, dtype=torch.float32)
        assert not is_float8tensor(large_tensor), "大张量不应该被识别为FP8张量"
        assert not is_mxfp8tensor(large_tensor), "大张量不应该被识别为MXFP8张量"
        
        # 测试稀疏张量
        sparse_tensor = torch.sparse_coo_tensor(torch.tensor([[0, 1], [2, 0]]), torch.tensor([3, 4]))
        assert not is_float8tensor(sparse_tensor), "稀疏张量不应该被识别为FP8张量"
        assert not is_mxfp8tensor(sparse_tensor), "稀疏张量不应该被识别为MXFP8张量"