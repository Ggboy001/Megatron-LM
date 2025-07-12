
import pytest

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_align_size


class TestFP8AlignSize:
    """测试FP8对齐大小函数"""

    @pytest.mark.internal
    def test_get_fp8_align_size_mxfp8(self):
        """测试MXFP8配方的对齐大小"""
        # MXFP8应该返回32字节对齐
        align_size = get_fp8_align_size(Fp8Recipe.mxfp8)
        assert align_size == 32, f"MXFP8对齐大小应该是32，但得到{align_size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_delayed(self):
        """测试延迟缩放配方的对齐大小"""
        # 延迟缩放应该返回16字节对齐
        align_size = get_fp8_align_size(Fp8Recipe.delayed)
        assert align_size == 16, f"延迟缩放对齐大小应该是16，但得到{align_size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_tensorwise(self):
        """测试张量级缩放配方的对齐大小"""
        # 张量级缩放应该返回16字节对齐
        align_size = get_fp8_align_size(Fp8Recipe.tensorwise)
        assert align_size == 16, f"张量级缩放对齐大小应该是16，但得到{align_size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_blockwise(self):
        """测试块级缩放配方的对齐大小"""
        # 块级缩放应该返回16字节对齐
        align_size = get_fp8_align_size(Fp8Recipe.blockwise)
        assert align_size == 16, f"块级缩放对齐大小应该是16，但得到{align_size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_all_recipes(self):
        """测试所有FP8配方的对齐大小"""
        # 测试所有已知的FP8配方
        recipes = [
            Fp8Recipe.delayed,
            Fp8Recipe.tensorwise,
            Fp8Recipe.blockwise,
            Fp8Recipe.mxfp8
        ]
        
        for recipe in recipes:
            align_size = get_fp8_align_size(recipe)
            
            # 验证对齐大小是有效的
            assert align_size in [16, 32], f"对齐大小应该是16或32，但得到{align_size}"
            
            # 验证对齐大小是2的幂次
            assert (align_size & (align_size - 1)) == 0, f"对齐大小应该是2的幂次，但{align_size}不是"
            
            # 验证对齐大小是正数
            assert align_size > 0, f"对齐大小应该是正数，但得到{align_size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_memory_alignment(self):
        """测试对齐大小在内存分配中的实际应用"""
        # 模拟使用对齐大小进行内存分配
        for recipe in [Fp8Recipe.delayed, Fp8Recipe.mxfp8]:
            align_size = get_fp8_align_size(recipe)
            
            # 创建不同大小的张量，验证对齐
            test_sizes = [64, 128, 256, 512, 1024]
            
            for size in test_sizes:
                # 模拟分配内存
                allocated_size = (size + align_size - 1) // align_size * align_size
                
                # 验证分配的大小是对齐的
                assert allocated_size % align_size == 0, \
                    f"分配的大小{alignated_size}应该对齐到{align_size}"
                
                # 验证分配的大小不小于原始大小
                assert allocated_size >= size, \
                    f"分配的大小{alignated_size}应该不小于原始大小{size}"

    @pytest.mark.internal
    def test_get_fp8_align_size_edge_cases(self):
        """测试边界情况"""
        # 测试无效的配方（如果将来添加了新的配方）
        # 这里我们测试当前已知的配方是否都能正确处理
        
        recipes = [
            Fp8Recipe.delayed,
            Fp8Recipe.tensorwise, 
            Fp8Recipe.blockwise,
            Fp8Recipe.mxfp8
        ]
        
        for recipe in recipes:
            try:
                align_size = get_fp8_align_size(recipe)
                # 如果没有抛出异常，验证返回值
                assert isinstance(align_size, int), f"对齐大小应该是整数，但得到{type(align_size)}"
                assert align_size > 0, f"对齐大小应该是正数，但得到{align_size}"
            except Exception as e:
                pytest.fail(f"配方{recipe}应该能正确处理，但抛出了异常: {e}")

    @pytest.mark.internal
    def test_get_fp8_align_size_consistency(self):
        """测试函数的一致性"""
        # 多次调用应该返回相同的结果
        for recipe in [Fp8Recipe.delayed, Fp8Recipe.mxfp8]:
            align_size_1 = get_fp8_align_size(recipe)
            align_size_2 = get_fp8_align_size(recipe)
            align_size_3 = get_fp8_align_size(recipe)
            
            assert align_size_1 == align_size_2 == align_size_3, \
                f"多次调用应该返回相同的结果: {align_size_1}, {align_size_2}, {align_size_3}"

    @pytest.mark.internal
    def test_get_fp8_align_size_documentation_consistency(self):
        """测试函数行为与文档的一致性"""
        # 根据函数实现，MXFP8返回32，其他返回16
        # 这个测试验证函数行为是否符合预期
        
        # MXFP8应该返回32
        mxfp8_align = get_fp8_align_size(Fp8Recipe.mxfp8)
        assert mxfp8_align == 32, "MXFP8应该返回32字节对齐"
        
        # 其他配方应该返回16
        other_recipes = [Fp8Recipe.delayed, Fp8Recipe.tensorwise, Fp8Recipe.blockwise]
        for recipe in other_recipes:
            align_size = get_fp8_align_size(recipe)
            assert align_size == 16, f"配方{recipe}应该返回16字节对齐，但得到{align_size}"