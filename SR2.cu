#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h>

#define N 10000

// ядро
__global__ void calculateRiemannZeta(double* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        double sum = 0.0;
        for (int i = 1; i <= N; i++) { // Изменено до N
            sum += 1.0 / (double)(i * i);
        }
        result[idx] = sum;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
//a. Доработайте задание по выводу параметров видео-карты, добавив в него вывод объема константной памяти и пиковую частоту видеокарты в МГц.---------------------------------------------------------------------------
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);//определение параметров GPU с номером 0
        printf("Device name : %s\n", deviceProp.name);
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

        //new
        printf("a) Total constant memory: %lu\n", deviceProp.totalConstMem);
        printf("    Peak clock frequency: %d MHz\n\n", deviceProp.clockRate / 1000);

        scanf("");
    }

 //b. Напишите программу вычисления дзета-функции Римана путем суммирования ряда из обратных степеней.-------------------------------------------------------------------
    double* d_result, * h_result;
    double zeta;

    //выделение памяти на устройстве и хосте
    cudaMalloc((void**)&d_result, sizeof(double));
    h_result = (double*)malloc(sizeof(double));

    //запуск ядра
    calculateRiemannZeta << <1, 1 >> > (d_result);

    //копирование результата на хост
    cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    zeta = *h_result;
    printf("Value of Riemann Zeta function: %f\n", zeta);

    //освобождение памяти
    free(h_result);
    cudaFree(d_result);

    //Проверка на ошибки выполнения функции
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}