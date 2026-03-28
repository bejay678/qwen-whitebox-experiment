#!/bin/bash
# C适配器编译脚本

echo "=== 编译C适配器 ==="
echo "开始时间: $(date)"

# 检查编译器
echo "1. 检查编译器..."
gcc --version | head -1
echo ""

# 编译选项
CFLAGS="-O3 -march=native -fPIC -Wall -Wextra"
LDFLAGS="-shared -lm"

echo "2. 编译适配器库..."
gcc $CFLAGS -c adapter.c -o adapter.o
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo "3. 链接共享库..."
gcc $LDFLAGS adapter.o -o libadapter.so
if [ $? -ne 0 ]; then
    echo "❌ 链接失败"
    exit 1
fi

echo "4. 编译测试程序..."
if [ -f test_adapter.c ]; then
    gcc $CFLAGS test_adapter.c -o test_adapter -L. -ladapter
    echo "测试程序编译完成"
fi

echo "5. 验证库文件..."
if [ -f libadapter.so ]; then
    file libadapter.so
    ls -lh libadapter.so
    echo "✅ 库文件创建成功"
else
    echo "❌ 库文件未创建"
    exit 1
fi

echo ""
echo "=== 编译完成 ==="
echo "结束时间: $(date)"
echo ""
echo "📋 编译信息:"
echo "  优化级别: O3"
echo "  架构优化: -march=native"
echo "  库类型: 共享库 (.so)"
echo "  输出文件: libadapter.so"
echo ""
echo "🚀 使用说明:"
echo "  1. 设置库路径: export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH"
echo "  2. Python调用: python ../scripts/c_adapter_wrapper.py"
echo "  3. C程序调用: ./test_adapter (如果编译了测试程序)"
echo ""
echo "🎉 C适配器编译成功！"