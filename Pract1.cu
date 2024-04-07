#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h>

#define N 10000000

// ядро
__global__ void add(int* a, int* b, int* c) {
    *c = *a + *b;
}
__global__ void copyData(float* input, float* output)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N)
    {
        output[index] = input[index];
    }
}
__global__ void calculatePi(double* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        double x = (double)idx / N;
        double y = sqrtf(1 - x * x); // Формула окружности x^2 + y^2 = r^2

        result[idx] = y / N;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
//b.Напишите программу, которая выводит на экран результат сложения двух чисел.-----------------------------------------------------------------------------------
    int a, b, c;

    // переменные на GPU
    int* dev_a, * dev_b, * dev_c;
    int size = sizeof(int); //размерность
    // выделяем память на GPU
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    // инициализация переменных
    a = 2;
    b = 4;
    // копирование информации с CPU на GPU
    cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice
    );
    cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice
    );
    // вызов ядра
    add << < 1, 1 >> > (dev_a, dev_b, dev_c);
    // копирование результата работы ядра с GPU на CPU
        cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
    // вывод информации
    
    printf("b) %d + %d = %d\n\n", a, b, c);
    // очищение памяти на GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

//c. Напишите программу для определения следующих параметров видеокарты с поддержкой технологии CUDA---------------------------------------------------------------------------
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);//определение параметров GPU с номером 0
        printf("c) Device name : %s\n", deviceProp.name);
        printf("Total global memory : %d MB\n",
            deviceProp.totalGlobalMem / 1024 / 1024);
        printf("Shared memory per block : %d\n",
            deviceProp.sharedMemPerBlock);
        printf("Registers per block : %d\n",
            deviceProp.regsPerBlock);
        printf("Warp size : %d\n", deviceProp.warpSize);
        printf("Memory pitch : %d\n", deviceProp.memPitch);
        printf("Max threads per block : %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("Max threads dimensions : x = %d, y = %d, z = % d\n", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Max grid size: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        printf("Clock rate: %d\n", deviceProp.clockRate);
        printf("Total constant memory: %d\n",
            deviceProp.totalConstMem);
        printf("Compute capability: %d.%d\n",
            deviceProp.major, deviceProp.minor);
        printf("Texture alignment: %d\n",
            deviceProp.textureAlignment);
        printf("Device overlap: %d\n",
            deviceProp.deviceOverlap);
        printf("Multiprocessor count: %d\n",
            deviceProp.multiProcessorCount);
        printf("Kernel execution timeout enabled: %s\n\n",
            deviceProp.kernelExecTimeoutEnabled ? "true" :
            "false");
        scanf("");
    }

//d. Измерьте скорость копирования данных (ГБ/сек) между CPU и GPU.-------------------------------------------------------------------------------------
    //#define N 10000000

    float* h_input = new float[N];
    float* h_output = new float[N];
    float* d_input, * d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_input[i] = i;
    }

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;
    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    copyData << <numBlocks, blockSize >> > (d_input, d_output);
    // вызов ядра
    cudaEventRecord(stop, 0);
    // ожидание завершения работы ядра
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // вывод информации 
    printf("d) Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
    double dataSize = N * sizeof(float);
    double dataRate = dataSize / (elapsedTime*1e6);
    printf("Data transfer rate: %f GB/sec\n\n", dataRate);
        // уничтожение события 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

// e.Напишите программу, вычисляющую число пи методом интегрирования четверти окружности единичного радиуса
// (можно использовать формулу для вычисления площади четверти круга.В программе предусмотрите проверку на ошибку выполнения функции.-----------------------------------------
    printf("e) Write a program to calculate pi by integrating a quarter circle of unit radius\n");
    double* d_result, * h_result;
    double pi;

    // Выделение памяти на устройстве и хосте
    cudaMalloc((void**)&d_result, N * sizeof(double));
    h_result = (double*)malloc(N * sizeof(double));

    // Запуск ядра
    //blockSize = 256;
    //numBlocks = (N + blockSize - 1) / blockSize;
    calculatePi << <numBlocks, blockSize >> > (d_result);

    // Копирование результатов на хост
    cudaMemcpy(h_result, d_result, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Суммирование результатов
    pi = 0;
    for (int i = 0; i < N; i++) {
        pi += h_result[i];
    }

    // Умножаем на 4 для получения значения π
    pi *= 4;

    printf("Estimated value of pi: %f\n", pi);

    // Освобождение памяти
    free(h_result);
    cudaFree(d_result);

    // Проверка на ошибки выполнения функции
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }


    return 0;
}
