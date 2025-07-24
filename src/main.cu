#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels.h"

const int input_size = 2;
const int hidden_size = 4;
const int output_size = 1;
const int epochs = 20000;
const float learning_rate = 0.05f;

const float h_inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
const float h_targets[4] = { 0, 1, 1, 0 };

float* d_inputs, * d_hidden, * d_output, * d_target;
float* d_weights1, * d_bias1, * d_weights2, * d_bias2;


int main()
{
    cudaMalloc(&d_inputs, sizeof(float) * input_size);
    cudaMalloc(&d_hidden, sizeof(float) * hidden_size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_target, sizeof(float));
    cudaMalloc(&d_weights1, sizeof(float) * input_size * hidden_size);
    cudaMalloc(&d_bias1, sizeof(float) * hidden_size);
    cudaMalloc(&d_weights2, sizeof(float) * hidden_size * output_size);
    cudaMalloc(&d_bias2, sizeof(float) * output_size);

    float h_w1[input_size * hidden_size];
    float h_b1[hidden_size] = {};
    float h_w2[hidden_size * output_size];
    float h_b2[output_size] = {};
    srand(42);
    for (int i = 0; i < input_size * hidden_size; i++) h_w1[i] = (rand() / (float)RAND_MAX - 0.5f);
    for (int i = 0; i < hidden_size * output_size; i++) h_w2[i] = (rand() / (float)RAND_MAX - 0.5f);

    cudaMemcpy(d_weights1, h_w1, sizeof(h_w1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, h_b1, sizeof(h_b1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, h_w2, sizeof(h_w2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, h_b2, sizeof(h_b2), cudaMemcpyHostToDevice);


    // Training
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < 4; i++)
        {
            cudaMemcpy(d_inputs, h_inputs[i], sizeof(float) * input_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, &h_targets[i], sizeof(float), cudaMemcpyHostToDevice);

            forwardPass << <1, hidden_size >> > (d_inputs, d_weights1, d_bias1, d_hidden, input_size, hidden_size);
            cudaDeviceSynchronize();

            forwardPass<< <1, output_size >> > (d_hidden, d_weights2, d_bias2, d_output, hidden_size, output_size);
            cudaDeviceSynchronize();

            backPropagation << <1, hidden_size >> > (d_inputs, d_hidden, d_output,
                d_weights1, d_bias1, d_weights2, d_bias2,
                d_target,
                input_size, hidden_size, learning_rate);
            cudaDeviceSynchronize();
        }
    }

    // Inference
    printf("=== Inference After Training ===\n");
    for (int i = 0; i < 4; i++)
    {
        cudaMemcpy(d_inputs, h_inputs[i], sizeof(float) * input_size, cudaMemcpyHostToDevice);
        forwardPass<< <1, hidden_size >> > (d_inputs, d_weights1, d_bias1, d_hidden, input_size, hidden_size);
        cudaDeviceSynchronize();
        forwardPass<< <1, output_size >> > (d_hidden, d_weights2, d_bias2, d_output, hidden_size, output_size);
        cudaDeviceSynchronize();

        float out;
        cudaMemcpy(&out, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%d XOR %d = %.4f (target=%.1f)\n",
            (int)h_inputs[i][0], (int)h_inputs[i][1], out, h_targets[i]);
    }

    cudaFree(d_inputs); cudaFree(d_hidden); cudaFree(d_output); cudaFree(d_target);
    cudaFree(d_weights1); cudaFree(d_bias1); cudaFree(d_weights2); cudaFree(d_bias2);
    
    
    return 0;
}