import ctypes
import numpy as np
from pathlib import Path

class CAdapter:
    """C适配器Python包装器"""
    
    def __init__(self, lib_path=None, weight_dir=None):
        if lib_path is None:
            lib_path = "/root/千问白盒化实验/c_adapter/libadapter.so"
        if weight_dir is None:
            weight_dir = "/root/千问白盒化实验/c_adapter"
        
        print(f"初始化C适配器: {lib_path}")
        
        # 加载共享库
        try:
            self.lib = ctypes.CDLL(lib_path)
            print("✅ C适配器库加载成功")
        except Exception as e:
            print(f"❌ C适配器库加载失败: {e}")
            raise
        
        # 定义函数原型
        self.lib.adapter_init.argtypes = [ctypes.c_char_p]
        self.lib.adapter_init.restype = ctypes.c_int
        
        self.lib.adapter_forward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        ]
        self.lib.adapter_forward.restype = None
        
        self.lib.adapter_get_dims.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.adapter_get_dims.restype = None
        
        self.lib.adapter_cleanup.argtypes = []
        self.lib.adapter_cleanup.restype = None
        
        # 初始化适配器
        ret = self.lib.adapter_init(weight_dir.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"C适配器初始化失败，错误码: {ret}")
        
        # 获取维度
        self.input_dim = ctypes.c_int()
        self.hidden_dim = ctypes.c_int()
        self.output_dim = ctypes.c_int()
        
        self.lib.adapter_get_dims(
            ctypes.byref(self.input_dim),
            ctypes.byref(self.hidden_dim),
            ctypes.byref(self.output_dim)
        )
        
        self.input_dim = self.input_dim.value
        self.hidden_dim = self.hidden_dim.value
        self.output_dim = self.output_dim.value
        
        print(f"✅ C适配器初始化成功: {self.input_dim} → {self.hidden_dim} → {self.output_dim}")
    
    def forward(self, input_vector):
        """前向传播"""
        if input_vector.shape != (self.input_dim,):
            raise ValueError(f"输入向量形状应为({self.input_dim},)，实际为{input_vector.shape}")
        
        output = np.zeros(self.output_dim, dtype=np.float32)
        self.lib.adapter_forward(input_vector.astype(np.float32), output)
        return output
    
    def forward_batch(self, input_batch):
        """批量前向传播"""
        if len(input_batch.shape) != 2 or input_batch.shape[1] != self.input_dim:
            raise ValueError(f"输入批次形状应为(N, {self.input_dim})，实际为{input_batch.shape}")
        
        batch_size = input_batch.shape[0]
        output_batch = np.zeros((batch_size, self.output_dim), dtype=np.float32)
        
        for i in range(batch_size):
            self.lib.adapter_forward(input_batch[i].astype(np.float32), output_batch[i])
        
        return output_batch
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'lib'):
            self.lib.adapter_cleanup()
            print("C适配器资源已清理")

def test_c_adapter():
    """测试C适配器"""
    print("=== 测试C适配器 ===")
    
    try:
        adapter = CAdapter()
        
        # 生成测试输入
        np.random.seed(42)
        test_input = np.random.randn(adapter.input_dim).astype(np.float32)
        
        print(f"测试输入形状: {test_input.shape}")
        
        # 单次前向
        output = adapter.forward(test_input)
        print(f"输出形状: {output.shape}")
        print(f"输出前5个值: {output[:5]}")
        
        # 批量测试
        batch_size = 3
        batch_input = np.random.randn(batch_size, adapter.input_dim).astype(np.float32)
        batch_output = adapter.forward_batch(batch_input)
        
        print(f"批量输入形状: {batch_input.shape}")
        print(f"批量输出形状: {batch_output.shape}")
        
        print("✅ C适配器测试通过")
        return adapter
        
    except Exception as e:
        print(f"❌ C适配器测试失败: {e}")
        return None

if __name__ == "__main__":
    test_c_adapter()