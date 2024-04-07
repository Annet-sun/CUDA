#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <device_atomic_functions.h>

#define N 10000
//#define N 100000000 //для b

//------------------------------------------------------------A------------------------------------------------------------------------------
// Функция для генерации случайного числа в диапазоне [0, 1]
__device__ float random_float(curandState* state) {
    return curand_uniform(state) + 1.0e-7;
}

__global__ void monteCarloPi(unsigned int* count, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[tid];
    unsigned int pointsInsideCircle = 0;

    // Генерация одной случайной точки и подсчет, попала ли она внутрь круга
    for (int i = 0; i < N; ++i) {
        float x = random_float(&localState);
        float y = random_float(&localState);
        float distance = x * x + y * y;
        if (distance <= 1.0f)
            pointsInsideCircle++;
    }

    atomicAdd(count, pointsInsideCircle);
}

__global__ void setup_kernel(curandState* states, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}
//------------------------------------------------------------B------------------------------------------------------------------------------

#define BASE_TYPE float
#define MAX_GRIDSIZE 1


__global__ void expMass(BASE_TYPE* A, int arraySize) {
    int index = (blockIdx.y * MAX_GRIDSIZE + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < arraySize) {
        A[index] = expf((BASE_TYPE)((index % 360) * M_PI / 180));
    }
}


//------------------------------------------------------------C------------------------------------------------------------------------------
__global__ void dotProduct(float* a, float* b, float* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = tid; i < N; i += stride) {
        sum += a[i] * b[i];
    }

    atomicAdd(c, sum);
}

//------------------------------------------------------------D------------------------------------------------------------------------------
__global__ void isOrthogonal(int* a, int* result, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[row * n + k] * a[j * n + k];
            }
            if (row == j && sum != 1) {
                *result = 0;
                return;
            }
            if (row != j && sum != 0) {
                *result = 0;
                return;
            }
        }
    }
}

//---------------------------------------------------------MAIN--------------------------------------------------------------------------
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    dim3 threadsPerBlock(256); // Один блок содержит 256 потоков
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

//a. Вычислите приблизительное значение числа π методом Монте Карло. Постройте равномерную сетку нитей в квадрате [0, 1] на [0, 1] 
// и посчитайте количество нитей, которые попали в круг с центром в нуле и с радиусом, равным единице. 
// Для вычисления случайного числа напишите __device__ функцию.------------------------------------------------------------------------------------------
 
    //задаем зерно для генерации случайных чисел
    unsigned long seed = 1234;

    //выделение памяти на хосте и устройстве для подсчета количества точек в круге
    unsigned int* d_count;
    unsigned int h_count = 0;
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMemcpy(d_count, &h_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    //выделение памяти для генераторов случайных чисел
    curandState* devStates;
    cudaMalloc((void**)&devStates, N * sizeof(curandState));

    //инициализация генераторов
    setup_kernel << <(N + 255) / 256, 256 >> > (devStates, seed);



    //запуск ядра
    monteCarloPi << <numBlocks, threadsPerBlock >> > (d_count, devStates);

    //копирование результата с устройства на хост
    cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //освобождение памяти на устройстве
    cudaFree(d_count);
    cudaFree(devStates);

    //вычисление приближенного значения числа π
    float pi = 4.0f * (float)h_count / (float)(N * numBlocks.x * threadsPerBlock.x);
    printf("a) Approximate value of pi: %f\n\n", pi);

//b. Используя другие функции вместо синуса (__expf (e^x), __exp10f (10^x) и т.д.), 
// рассчитайте относительную ошибку (отношение абсолютной ошибки к значению функции).------------------------------------------------------------------------------------
    BASE_TYPE* h_A, * d_A;
    size_t size = N * sizeof(BASE_TYPE);

    //выделение памяти на хосте
    h_A = (BASE_TYPE*)malloc(size);

    //выделение памяти на устройстве
    cudaMalloc((void**)&d_A, size);

    //запуск ядра
    //dim3 threadsPerBlock(256);
    //dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    expMass << <numBlocks, threadsPerBlock >> > (d_A, N);

    //копирование данных с устройства на хост
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    //вычисление ошибки
    BASE_TYPE sum_err = 0;
    for (int i = 0; i < N; ++i) {
        BASE_TYPE exact = exp((BASE_TYPE)((i % 360) * 3.14159265358979323846 / 180));
        sum_err += fabs(exact - h_A[i]);
    }
    BASE_TYPE err = sum_err / N;

    printf("b) Relative error: %f\n\n", err);

    //освобождение памяти
    free(h_A);
    cudaFree(d_A);

//c. Напишите программу, реализующую скалярное произведение двух векторов.--------------------------------------------------------------------------------------
    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    size_t hostSize = N * sizeof(float);

    //выделение памяти на хосте
    h_a = (float*)malloc(hostSize);
    h_b = (float*)malloc(hostSize);
    h_c = (float*)malloc(sizeof(float));

    //инициализация векторов
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //выделение памяти на устройстве
    cudaMalloc((void**)&d_a, hostSize);
    cudaMalloc((void**)&d_b, hostSize);
    cudaMalloc((void**)&d_c, sizeof(float));

    //копирование данных с хоста на устройство
    cudaMemcpy(d_a, h_a, hostSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, hostSize, cudaMemcpyHostToDevice);

    //запуск ядра
    dotProduct << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c);

    //копирование результата с устройства на хост
    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    printf("c) Dot product: %f\n\n", *h_c);

    //освобождение памяти
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

//d. Проверить, является ли данная квадратная матрица ортогональной, путем умножения на транспонированную 
// (не выделяя под нее дополнительной памяти) и сравнения результата с единичной матрицей..---------------------------------------------------------------
    
    int* a, * dev_a, * d_result;
    int h_result = 1;

    //выделение памяти на хосте для матрицы a
    a = (int*)malloc(N * N * sizeof(int));

    //инициализация матрицы a (ортогональная)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                a[i * N + j] = 1;
            }
            else {
                a[i * N + j] = 0;
            }
        }
    }

    //инициализация матрицы a (НЕ ортогональная)
    //for (int i = 0; i < N * N; ++i) {
    //    a[i] = i + 1; // Просто заполним матрицу числами от 1 до N*N
    //}

    //выделение памяти на устройстве
    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    //копирование данных с хоста на устройство
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);

    //устанавливаем результат в 1 на устройстве
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    //запуск ядра
    isOrthogonal << <numBlocks, threadsPerBlock >> > (dev_a, d_result, N);

    //копирование результата с устройства на хост
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_result == 1) {
        printf("d) Matrix A is orthogonal.\n");
    }
    else {
        printf("Matrix A is not orthogonal.\n");
    }

    //освобождение памяти
    free(a);
    cudaFree(dev_a);
    cudaFree(d_result);

    //проверка на ошибки выполнения функции
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    return 0;
}


