// adapter.c - 适配器C语言实现
// 输入维度: 896
// 隐藏维度: 256
// 输出维度: 128
// 架构: 896 → 256 (ReLU) → 128 (LayerNorm)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 全局变量：权重、偏置、维度
static float *fc1_weight = NULL;  // 第一层权重 (256×896)
static float *fc1_bias = NULL;    // 第一层偏置 (256)
static float *fc2_weight = NULL;  // 第二层权重 (128×256)
static float *fc2_bias = NULL;    // 第二层偏置 (128)
static float *ln_weight = NULL;   // LayerNorm权重 (128)
static float *ln_bias = NULL;     // LayerNorm偏置 (128)

static int input_dim = 0;
static int hidden_dim = 0;
static int output_dim = 0;

// ReLU激活函数
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

// 初始化：从二进制文件加载权重和偏置
int adapter_init(const char *weight_dir) {
    char filepath[256];
    FILE *f;
    
    // 读取维度
    snprintf(filepath, sizeof(filepath), "%s/adapter_dims.txt", weight_dir);
    f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "无法打开维度文件: %s\n", filepath);
        return -1;
    }
    fscanf(f, "%d %d %d", &input_dim, &hidden_dim, &output_dim);
    fclose(f);
    
    printf("适配器初始化: %d → %d → %d\n", input_dim, hidden_dim, output_dim);
    
    // 分配内存
    fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    fc1_bias = (float*)malloc(hidden_dim * sizeof(float));
    fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    fc2_bias = (float*)malloc(output_dim * sizeof(float));
    ln_weight = (float*)malloc(output_dim * sizeof(float));
    ln_bias = (float*)malloc(output_dim * sizeof(float));
    
    if (!fc1_weight || !fc1_bias || !fc2_weight || !fc2_bias || !ln_weight || !ln_bias) {
        fprintf(stderr, "内存分配失败\n");
        adapter_cleanup();
        return -1;
    }
    
    // 加载第一层权重
    snprintf(filepath, sizeof(filepath), "%s/fc1_weight.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开权重文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(fc1_weight, sizeof(float), hidden_dim * input_dim, f);
    fclose(f);
    
    // 加载第一层偏置
    snprintf(filepath, sizeof(filepath), "%s/fc1_bias.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开偏置文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(fc1_bias, sizeof(float), hidden_dim, f);
    fclose(f);
    
    // 加载第二层权重
    snprintf(filepath, sizeof(filepath), "%s/fc2_weight.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开权重文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(fc2_weight, sizeof(float), output_dim * hidden_dim, f);
    fclose(f);
    
    // 加载第二层偏置
    snprintf(filepath, sizeof(filepath), "%s/fc2_bias.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开偏置文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(fc2_bias, sizeof(float), output_dim, f);
    fclose(f);
    
    // 加载LayerNorm权重
    snprintf(filepath, sizeof(filepath), "%s/ln_weight.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开LayerNorm权重文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(ln_weight, sizeof(float), output_dim, f);
    fclose(f);
    
    // 加载LayerNorm偏置
    snprintf(filepath, sizeof(filepath), "%s/ln_bias.bin", weight_dir);
    f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "无法打开LayerNorm偏置文件: %s\n", filepath);
        adapter_cleanup();
        return -1;
    }
    fread(ln_bias, sizeof(float), output_dim, f);
    fclose(f);
    
    printf("✅ 适配器初始化成功\n");
    printf("   参数: %d → %d → %d\n", input_dim, hidden_dim, output_dim);
    printf("   总权重: %.2f MB\n", 
           (hidden_dim * input_dim + hidden_dim + output_dim * hidden_dim + output_dim * 3) * sizeof(float) / 1024.0 / 1024.0);
    
    return 0;
}

// LayerNorm函数
static void layer_norm(float *x, const float *weight, const float *bias, int dim) {
    // 计算均值
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += x[i];
    }
    mean /= dim;
    
    // 计算方差
    float variance = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= dim;
    
    // 归一化并应用缩放和偏移
    float std_inv = 1.0f / sqrtf(variance + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) * std_inv * weight[i] + bias[i];
    }
}

// 前向传播：input 长度为 input_dim，output 长度为 output_dim
void adapter_forward(const float *input, float *output) {
    // 第一层: input_dim → hidden_dim
    float hidden[hidden_dim];
    
    for (int i = 0; i < hidden_dim; ++i) {
        float sum = fc1_bias[i];
        const float *w_row = fc1_weight + i * input_dim;
        
        for (int j = 0; j < input_dim; ++j) {
            sum += w_row[j] * input[j];
        }
        hidden[i] = relu(sum);
    }
    
    // 第二层: hidden_dim → output_dim
    for (int i = 0; i < output_dim; ++i) {
        float sum = fc2_bias[i];
        const float *w_row = fc2_weight + i * hidden_dim;
        
        for (int j = 0; j < hidden_dim; ++j) {
            sum += w_row[j] * hidden[j];
        }
        output[i] = sum;
    }
    
    // LayerNorm
    layer_norm(output, ln_weight, ln_bias, output_dim);
}

// 批量前向传播：inputs 形状为 (batch_size, input_dim)，outputs 形状为 (batch_size, output_dim)
void adapter_forward_batch(const float *inputs, float *outputs, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        const float *input = inputs + b * input_dim;
        float *output = outputs + b * output_dim;
        adapter_forward(input, output);
    }
}

// 获取维度信息
void adapter_get_dims(int *in_dim, int *hid_dim, int *out_dim) {
    if (in_dim) *in_dim = input_dim;
    if (hid_dim) *hid_dim = hidden_dim;
    if (out_dim) *out_dim = output_dim;
}

// 清理
void adapter_cleanup() {
    if (fc1_weight) free(fc1_weight);
    if (fc1_bias) free(fc1_bias);
    if (fc2_weight) free(fc2_weight);
    if (fc2_bias) free(fc2_bias);
    if (ln_weight) free(ln_weight);
    if (ln_bias) free(ln_bias);
    
    fc1_weight = fc1_bias = fc2_weight = fc2_bias = ln_weight = ln_bias = NULL;
    input_dim = hidden_dim = output_dim = 0;
    
    printf("适配器清理完成\n");
}

// 测试函数
void test_adapter() {
    if (input_dim == 0) {
        printf("适配器未初始化\n");
        return;
    }
    
    // 创建测试输入
    float *input = (float*)malloc(input_dim * sizeof(float));
    float *output = (float*)malloc(output_dim * sizeof(float));
    
    if (!input || !output) {
        printf("测试内存分配失败\n");
        free(input);
        free(output);
        return;
    }
    
    // 初始化输入（简单测试）
    for (int i = 0; i < input_dim; i++) {
        input[i] = 0.1f;
    }
    
    // 运行适配器
    adapter_forward(input, output);
    
    printf("适配器测试完成\n");
    printf("输入维度: %d\n", input_dim);
    printf("输出维度: %d\n", output_dim);
    printf("输出前5个值: ");
    for (int i = 0; i < 5 && i < output_dim; i++) {
        printf("%.6f ", output[i]);
    }
    printf("\n");
    
    free(input);
    free(output);
}