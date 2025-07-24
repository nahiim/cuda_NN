#ifndef KERNELS_H
#define KERNELS_H

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
__device__ float sigmoid_derivative(float y)
{
    return y * (1.0f - y);
}



__global__ void forwardPass(float* inputs, float* weights, float* biases,
    float* outputs, int input_size, int output_size)
{
    int idx = threadIdx.x;
    if (idx >= output_size) return;

    float sum = biases[idx];
    for (int i = 0; i < input_size; i++)
    {
        sum += inputs[i] * weights[i * output_size + idx];
    }
    outputs[idx] = sigmoid(sum);
}


__global__ void backPropagation(
    float* input,
    float* hidden,        
    float* output,        
    float* weights1,      
    float* bias1,         
    float* weights2,      
    float* bias2,         
    float* target,        
    int input_size,
    int hidden_size,
    float learning_rate)
{
    int idx = threadIdx.x;

    if (idx >= hidden_size) return;

    float y = output[0];
    float t = target[0];

    float d_output = (y - t) * y * (1 - y);

    float h = hidden[idx];
    float grad_w2 = d_output * h;
    weights2[idx] -= learning_rate * grad_w2;

    if (idx == 0)
        bias2[0] -= learning_rate * d_output;

    float d_hidden = d_output * weights2[idx] * h * (1 - h);

    for (int i = 0; i < input_size; i++)
    {
        float x = input[i];
        float* w = &weights1[i * hidden_size + idx];
        *w -= learning_rate * d_hidden * x;
    }

    bias1[idx] -= learning_rate * d_hidden;
}



#endif
