#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <device_atomic_functions.h>
#include <cublas_v2.h>
#include <cublasxt.h>

#define N 1024
#define BLOCK_SIZE 256

//-----------------------------------------------------------------A-------------------------------------------------------------------------
//ядро для вычисления скалярного произведения с использованием разделяемой памяти
__global__ void dotProduct(float* a, float* b, float* c) {
    __shared__ float temp[N];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[tid] * b[tid];
    __syncthreads();

    //вычисление суммы элементов в разделяемой памяти
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}

//----------------------------------------------------------------B--------------------------------------------------------------------------

#define A 0.0   //начальная точка интервала интегрирования
#define B 1.0  //конечная точка интервала интегрирования 1 или  M_PI 
#define DX ((B - A) / N) //ширина каждого подынтервала

//функция, которая вычисляет значение подынтегральной функции в точке x
__device__ float f(float x) {
    return sin(x); // Пример: интеграл sin(x) от A до B
}

//ядро для вычисления определенного интеграла методом центральных прямоугольников
__global__ void integrate(float* result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float x = A + (idx + 0.5f) * DX; //вычисление центра текущего подынтервала
    atomicAdd(result, f(x) * DX);    //суммирование значения функции на текущем подынтервале
}

//----------------------------------------------------------------C--------------------------------------------------------------------------
__constant__ float c_a[N];
__constant__ float c_b[N];

//ядро для вычисления скалярного произведения с использованием константной памяти
__global__ void dotProduct_const_memory(float* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = tid; i < N; i += stride) {
        sum += c_a[i] * c_b[i];
    }

    atomicAdd(c, sum);
}

//----------------------------------------------------------------D--------------------------------------------------------------------------
//
////#define A 0.0   // Начальная точка интервала интегрирования
////#define B 1.0  // Конечная точка интервала интегрирования 1 или  M_PI 
////#define DX ((B - A) / N) // Ширина каждого подынтервала
//////Функция, которая вычисляет значение подынтегральной функции в точке x
//__host__ float f_sin(float x) {
//    return sin(x); // Пример: интеграл sin(x) от A до B
//}
//
////определение текстурной памяти
//texture<float, cudaTextureType1D, cudaReadModeElementType> texData;
//
////ядро для вычисления определенного интеграла методом центральных прямоугольников с использованием текстурной памяти
//__global__ void integrate_texture_memory(float* result) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    float x = A + (idx + 0.5f) * DX; // Вычисление центра текущего подынтервала
//    float value = tex1D(texData, x); // Получение значения функции из текстурной памяти
//    atomicAdd(result, value * DX); // Суммирование значения функции на текущем подынтервале
//}
//

//------------------------------------------------------------------E------------------------------------------------------------------------
void checkCudaErrors(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA API call failed with error: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
}

void checkCublasErrors(cublasStatus_t cublasStatus) {
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS API call failed with error code %d\n", cublasStatus);
        exit(EXIT_FAILURE);
    }
}

//---------------------------------------------------------MAIN-----------------------------------------------------------------------------------------
int main() {

    //a. Напишите программу для скалярного умножения двух векторов, используя разделяемую память. 
    // Разделяемая память выделяется с учетом ограничения ее размеров на блок. Количество нитей 
    // при вызове ядра равно числу элементов массива с разделяемой памятью. Замерьте время работы ядра программы.---------------------------------------
    float* h_a, * h_b, * h_c;  //векторы на хосте
    float* d_a, * d_b, * d_c;  //векторы на устройстве

    //выделение памяти на хосте
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(sizeof(float));

    //инициализация векторов на хосте
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //выделение памяти на устройстве
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, sizeof(float));

    //копирование данных из хоста на устройство
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    //запуск ядра
    dotProduct << <1, N >> > (d_a, d_b, d_c);

    //копирование результата с устройства на хост
    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    printf("a) Scalar dot product: %f\n", *h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //b. Напишите программу вычисления определенного интеграла функции одной переменной 
    // по квадратурной формуле центральных прямоугольников.------------------------------------------------------------------------------------

    float* d_result, h_result;//результат на устройстве и хосте

    //выделение памяти на устройстве для результата
    cudaMalloc((void**)&d_result, sizeof(float));
    //инициализация результата на устройстве
    cudaMemset(d_result, 0, sizeof(float));

    //вычисление количества блоков и потоков
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    //запуск ядра
    integrate << <numBlocks, threadsPerBlock >> > (d_result);

    //копирование результата с устройства на хост
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);


    //для функции sin(x) при интегрировании от 0 до 1 точный результат равен 1−cos(1 радиан), что примерно равно 0.4597
    printf("b) Result integral: %f\n", h_result);

    cudaFree(d_result);

    //c.Напишите программу для скалярного умножения двух векторов, используя константную память.-------------------------------------------------------
    // Замерьте время работы ядра программы.

    //выделение памяти на хосте
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(sizeof(float));

    //инициализация векторов на хосте
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //копирование данных в константную память
    cudaMemcpyToSymbol(c_a, h_a, N * sizeof(float));
    cudaMemcpyToSymbol(c_b, h_b, N * sizeof(float));

    //выделение памяти на устройстве для результата
    cudaMalloc((void**)&d_c, sizeof(float));
    cudaMemset(d_c, 0, sizeof(float));

    //инициализация событий CUDA для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //запуск ядра для вычисления скалярного произведения
    dotProduct_const_memory << <numBlocks, threadsPerBlock >> > (d_c);

    //ожидание завершения выполнения ядра
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //копирование результата с устройства на хост
    cudaMemcpy(h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    printf("c) Scalar product: %f\n", *h_c);
    printf("Time : %f milliseconds\n", milliseconds);

    //освобождение выделенной на хосте памяти
    free(h_a);
    free(h_b);
    //освобождение выделенной на устройстве памяти
    cudaFree(d_c);

    //d. Напишите программу вычисления определенного интеграла функции одной переменной по квадратурной формуле центральных прямоугольников.
    //  Используйте тектстурную память, привязанную к линейной памяти.d. Напишите программу вычисления определенного интеграла функции 
    // одной переменной по квадратурной формуле центральных прямоугольников. Используйте тектстурную память, привязанную к линейной памяти.----------

    //float* h_data_d;    //входные данные на хосте
    //float* d_result_d;  //результат на устройстве

    ////выделение памяти на хосте и заполнение входных данных
    //h_data_d = (float*)malloc(N * sizeof(float));
    //for (int i = 0; i < N; ++i) {
    //    h_data_d[i] = f_sin(A + i * ((B - A) / N)); // Заполнение массива значениями функции sin(x)
    //}

    ////выделение памяти на устройстве для результата
    //cudaMalloc((void**)&d_result_d, sizeof(float));
    //cudaMemset(d_result_d, 0, sizeof(float));

    ////копирование данных в текстурную память
    //cudaArray* cuArray;
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cudaMallocArray(&cuArray, &channelDesc, N, 1);
    //cudaMemcpyToArray(cuArray, 0, 0, h_data_d, N * sizeof(float), cudaMemcpyHostToDevice);
    //cudaBindTextureToArray(texData, cuArray, channelDesc);

    //// Запуск ядра для вычисления интеграла
    //integrate_texture_memory << <(N + 255) / 256, 256 >> > (d_result_d);

    ////копирование результата с устройства на хост
    //float h_result_d;
    //cudaMemcpy(&h_result_d, d_result_d, sizeof(float), cudaMemcpyDeviceToHost);

    //printf("Result integral: %f\n", h_result_d);

    ////освобождение памяти на хосте и устройстве
    //free(h_data_d);
    //cudaFree(d_result_d);
    //cudaFreeArray(cuArray);
    //cudaUnbindTexture(texData);

    //e. Дополните пример обработкой ошибок. Учитывайте, как ошибки при выделении памяти на CPU и GPU (функции CUDA), 
    // так и возвращаемый статус функций cuBLAS.

    float* h_data, * d_data;
    cublasHandle_t handle;
    cublasStatus_t cublasStatus;
    cudaError_t cudaStatus;

    //выделение памяти на хосте
    h_data = (float*)malloc(N * sizeof(float));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate memory on the host\n");
        exit(EXIT_FAILURE);
    }

    //выделение памяти на устройстве
    cudaStatus = cudaMalloc((void**)&d_data, N * sizeof(float));
    checkCudaErrors(cudaStatus);

    //инициализация cuBLAS
    cublasStatus = cublasCreate(&handle);
    checkCublasErrors(cublasStatus);


    //освобождение памяти и завершение работы с cuBLAS
    cublasDestroy(handle);
    free(h_data);
    cudaFree(d_data);

    //f. Вычислите матрицу С = (A + B) * (A * B), где A и B квадратные матрицы, используя библиотеку cuBlasXT
    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    cublasXtDeviceSelect(handle, 0); // Выбор устройства CUDA

    //размерность матриц
    int n = 3; //размер квадратных матриц

    //выделение памяти на хосте для матриц A, B и C
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    //инициализация данных матриц A и B (пример)
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = i;
        h_B[i] = i + 1;
    }

    //выделение памяти на устройстве для матриц A, B и C
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    //копирование данных с хоста на устройство
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    //операции с использованием cuBLASXT
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);

    //копирование результата с устройства на хост
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    //вывод результата
    printf("Matrix C:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    //освобождение памяти
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //завершение работы с cuBLASXT и освобождение ресурсов
    cublasXtDestroy(handle);

    return 0;
}


