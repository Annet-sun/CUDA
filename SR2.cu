#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <device_atomic_functions.h>

//#define N 1000000
//#define N 10 //для с
#define N 4 //для d
#define M 3 //для e

////----------------------------------------------------------------A--------------------------------------------------------------------------------
//функция для генерации случайного числа в диапазоне [0, 1]
__device__ float random_float(curandState* state) {
    return curand_uniform(state) + 1.0e-7;
}

__global__ void monteCarloPi(unsigned int* count, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[tid];
    unsigned int pointsInsideCircle = 0;

    //генерация одной случайной точки и подсчет, попала ли она внутрь круга
    for (int i = 0; i < N; ++i) {
        float x = random_float(&localState);
        float y = random_float(&localState);
        float distance = x * x + y * y;

        /*if (distance <= 1.0f)
            pointsInsideCircle++;*/

            //метод без if
        float d2 = distance - 1;
        int a = (d2 - abs(d2)) / (2 * d2);
        pointsInsideCircle = pointsInsideCircle + a;
    }

    atomicAdd(count, pointsInsideCircle);
}

__global__ void monteCarloPiOptimized(unsigned int* count, curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[tid];
    unsigned int localPointsInsideCircle = 0; //локальная переменная для накопления результатов

    //генерация одной случайной точки и подсчет, попала ли она внутрь круга
    for (int i = 0; i < N; ++i) {
        float x = random_float(&localState);
        float y = random_float(&localState);
        float distance = x * x + y * y;

        if (distance <= 1.0f)
            localPointsInsideCircle++;
    }

    atomicAdd(count, localPointsInsideCircle);
}

__global__ void setup_kernel(curandState* states, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

//----------------------------------------------------------------B----------------------------------------------------------------------------
//функция для скалярного произведения двух векторов типа float
__global__ void dotProduct_float(float* a, float* b, float* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = tid; i < N; i += stride) {
        sum += a[i] * b[i];
    }

    *c += sum;
}

//функция для скалярного произведения двух векторов типа double
__global__ void dotProduct_double(double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0;
    for (int i = tid; i < N; i += stride) {
        sum += a[i] * b[i];
    }

    *c += sum; 
}

//------------------------------------------------------------------C--------------------------------------------------------------------------
__global__ void gram_schmidt(float* vectors, float* ortho_vectors, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        for (int i = 0; i < idx; ++i) {
            float dot_product = 0.0f;
            for (int j = 0; j < n; ++j) {
                dot_product += ortho_vectors[i * n + j] * vectors[idx * n + j];
            }
            for (int j = 0; j < n; ++j) {
                vectors[idx * n + j] -= dot_product * ortho_vectors[i * n + j];
            }
        }

        //нормализация ортогонального вектора
        float norm = 0.0f;
        for (int j = 0; j < n; ++j) {
            norm += vectors[idx * n + j] * vectors[idx * n + j];
        }
        norm = sqrt(norm);
        for (int j = 0; j < n; ++j) {
            ortho_vectors[idx * n + j] = vectors[idx * n + j] / norm;
        }
    }
}

//------------------------------------------------------------------D--------------------------------------------------------------------------
//ядро для умножения матриц
__global__ void matrix_multiply(float* A, float* B, float* C, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

//------------------------------------------------------------------E--------------------------------------------------------------------------
__global__ void matrix_add(float* A, float* B, float* C, int rows, int cols) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        C[index] = A[index] + B[index];
    }
}


//---------------------------------------------------------MAIN-------------------------------------------------------------------------------------------
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    //a, b, c
    //dim3 threadsPerBlock(256); // Один блок содержит 256 потоков
    //dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);


    //a.Вызов атомарной функции, в целом, замедляет работу программы.Например, если очень много нитей должны добавить единицу одной-----------------------------
    // и той же переменной, то они встанут в очередь, и будут простаивать, пока все по очереди не выполнят эту операцию.
    // Одним из подходов для решения этой проблемы является уменьшение количества нитей, вызывающих атомарную операцию.
    // Учитывая выше сказанное, подумайте, как можно уменьшить количество нитей, вызывающих атомарную операцию в методе Монте Карло.
    // Оптимизируйте таким образом программу, и посмотрите, насколько уменьшается время ее работы.Также посмотрите, как меняется точность 
    // вычисления числа π.


    //Чтобы уменьшить количество нитей, вызывающих атомарную операцию в методе Монте Карло, можно использовать метод параллельного суммирования
    //Вместо того, чтобы каждая нить увеличивала счетчик отдельно, мы можем сначала накопить результаты всех нитей в локальных переменных, 
    //а затем выполнить одну атомарную операцию для суммирования этих локальных результатов.Это позволит значительно снизить конфликты 
    //при доступе к памяти при выполнении атомарных операций.

    ////выделение памяти на хосте и устройстве для подсчета количества точек в круге
    //unsigned int* d_count1, * d_count2;
    //unsigned int h_count1 = 0, h_count2 = 0;
    //cudaMalloc((void**)&d_count1, sizeof(unsigned int));
    //cudaMalloc((void**)&d_count2, sizeof(unsigned int));
    //cudaMemcpy(d_count1, &h_count1, sizeof(unsigned int), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_count2, &h_count2, sizeof(unsigned int), cudaMemcpyHostToDevice);

    ////выделение памяти для генераторов случайных чисел
    //curandState* devStates;
    //cudaMalloc((void**)&devStates, N * sizeof(curandState));

    ////инициализация генераторов
    //unsigned long seed = 1234;
    //setup_kernel << <(N + 255) / 256, 256 >> > (devStates, seed);

    ////замер времени выполнения
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //float totalElapsedTime1 = 0.0f;
    //float totalElapsedTime2 = 0.0f;
    //const int numIterations = 2;

    ////замер времени выполнения для неоптимизированного метода
    //for (int i = 0; i < numIterations + 1; ++i) {
    //    // Замер времени выполнения monteCarloPi
    //    cudaEventRecord(start, 0);
    //    monteCarloPi << <numBlocks, threadsPerBlock >> > (d_count1, devStates);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    float elapsedTime1;
    //    cudaEventElapsedTime(&elapsedTime1, start, stop);
    //    if (i != 0) totalElapsedTime1 += elapsedTime1;
    //}
    //float averageTime1 = totalElapsedTime1 / numIterations;

    ////вывод среднего времени выполнения для неоптимизированного метода
    //printf("Average time taken for monteCarloPi over %d iterations: %f ms\n", numIterations, averageTime1);


    ////замер времени выполнения для оптимизированного метода
    //for (int i = 0; i < numIterations + 1; ++i) {
    //    // Замер времени выполнения monteCarloPiOptimized
    //    cudaEventRecord(start, 0);
    //    monteCarloPiOptimized << <numBlocks, threadsPerBlock >> > (d_count2, devStates);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    float elapsedTime2;
    //    cudaEventElapsedTime(&elapsedTime2, start, stop);
    //    if (i != 0) totalElapsedTime2 += elapsedTime2;
    //}
    //float averageTime2 = totalElapsedTime2 / numIterations;

    ////вывод среднего времени выполнения для оптимизированного метода
    //printf("Average time taken for monteCarloPiOptimized over %d iterations: %f ms\n", numIterations, averageTime2);

    ////освобождение ресурсов
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);

    ////копирование результата с устройства на хост
    //cudaMemcpy(&h_count1, d_count1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&h_count2, d_count2, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    ////освобождение памяти на устройстве
    //cudaFree(d_count1);
    //cudaFree(d_count2);
    //cudaFree(devStates);

    ////вычисление приближенного значения числа π
    //float pi1 = 4.0f * (float)h_count1 / (float)(N * numBlocks.x * threadsPerBlock.x);
    //float pi2 = 4.0f * (float)h_count2 / (float)(N * numBlocks.x * threadsPerBlock.x);
    //printf("Approximate value of pi (monteCarloPi1): %f\n", pi1);
    //printf("Approximate value of pi (monteCarloPiOptimized): %f\n", pi2);


    //b. Сравните скорость и точность вычислений с типами float и double для функций из задания для практической работы-----------------------------------
    // (задание(Напишите программу, реализующую скалярное произведение двух векторов)). 
    // Какой тип для каких задач лучше использовать и почему? Сделайте вывод.

    ////выделение памяти на хосте для векторов
    //float* h_a_float, * h_b_float, * h_c_float;
    //double* h_a_double, * h_b_double, * h_c_double;
    //h_a_float = (float*)malloc(N * sizeof(float));
    //h_b_float = (float*)malloc(N * sizeof(float));
    //h_c_float = (float*)malloc(sizeof(float));
    //h_a_double = (double*)malloc(N * sizeof(double));
    //h_b_double = (double*)malloc(N * sizeof(double));
    //h_c_double = (double*)malloc(sizeof(double));

    ////инициализация векторов на хосте
    //for (int i = 0; i < N; i++) {
    //    h_a_float[i] = 1.0f;
    //    h_b_float[i] = 2.0f;
    //    h_a_double[i] = 1.0;
    //    h_b_double[i] = 2.0;
    //}

    ////выделение памяти на устройстве для векторов
    //float* d_a_float, * d_b_float, * d_c_float;
    //double* d_a_double, * d_b_double, * d_c_double;
    //cudaMalloc((void**)&d_a_float, N * sizeof(float));
    //cudaMalloc((void**)&d_b_float, N * sizeof(float));
    //cudaMalloc((void**)&d_c_float, sizeof(float));
    //cudaMalloc((void**)&d_a_double, N * sizeof(double));
    //cudaMalloc((void**)&d_b_double, N * sizeof(double));
    //cudaMalloc((void**)&d_c_double, sizeof(double));

    ////копирование данных с хоста на устройство
    //cudaMemcpy(d_a_float, h_a_float, N * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b_float, h_b_float, N * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_a_double, h_a_double, N * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b_double, h_b_double, N * sizeof(double), cudaMemcpyHostToDevice);

    ////создание CUDA events для замера времени выполнения
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //float elapsedTimeFloat, elapsedTimeDouble;
    //float totalElapsedTimeFloat = 0.0f;
    //float totalElapsedTimeDouble = 0.0f;

    ////выполнение нескольких итераций для вычисления среднего времени выполнения с типом float
    //for (int i = 0; i < 11; ++i) {
    //    cudaEventRecord(start, 0);
    //    dotProduct_float << <256, 256 >> > (d_a_float, d_b_float, d_c_float);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&elapsedTimeFloat, start, stop);
    //    if (i != 0) {
    //        totalElapsedTimeFloat += elapsedTimeFloat;
    //    }
    //}
    //float averageElapsedTimeFloat = totalElapsedTimeFloat / 10.0f;
    //printf("Average time taken for dotProduct_float: %f ms\n", averageElapsedTimeFloat);

    ////выполнение нескольких итераций для вычисления среднего времени выполнения с типом double
    //for (int i = 0; i < 11; ++i) {
    //    cudaEventRecord(start, 0);
    //    dotProduct_double << <256, 256 >> > (d_a_double, d_b_double, d_c_double);
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&elapsedTimeDouble, start, stop);
    //    if (i != 0) {
    //        totalElapsedTimeDouble += elapsedTimeDouble;
    //    }
    //}
    //float averageElapsedTimeDouble = totalElapsedTimeDouble / 10.0f;
    //printf("Average time taken for dotProduct_double: %f ms\n", averageElapsedTimeDouble);

    ////копирование результата с устройства на хост
    //cudaMemcpy(h_c_float, d_c_float, sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_c_double, d_c_double, sizeof(double), cudaMemcpyDeviceToHost);

    ////вывод результата скалярного произведения для типа float
    //printf("Scalar product (float): %f\n", *h_c_float);
    //// Вывод результата скалярного произведения для типа double
    //printf("Scalar product (double): %lf\n", *h_c_double);

    ////расчет средней точности (средней ошибки)
    //float expected_float = 2.0f * N; // Ожидаемый результат скалярного произведения для типа float
    //double expected_double = 2.0 * N; // Ожидаемый результат скалярного произведения для типа double

    ////вычисление средней ошибки для типа float
    //float error_float = fabsf(expected_float - *h_c_float);
    //float average_error_float = error_float / N;
    //printf("Average error (float): %f\n", average_error_float);

    ////вычисление средней ошибки для типа double
    //double error_double = fabs(expected_double - *h_c_double);
    //double average_error_double = error_double / N;
    //printf("Average error (double): %lf\n", average_error_double);

    ////освобождение памяти на хосте и устройстве
    //free(h_a_float);
    //free(h_b_float);
    //free(h_c_float);
    //free(h_a_double);
    //free(h_b_double);
    //free(h_c_double);
    //cudaFree(d_a_float);
    //cudaFree(d_b_float);
    //cudaFree(d_c_float);
    //cudaFree(d_a_double);
    //cudaFree(d_b_double);
    //cudaFree(d_c_double);
    ///*Вывод - Если нужна высокая точность, то лучше использовать тип double.
    //Однако, если точность не является критическим фактором, и скорость работы важнее, то можно использовать тип float*/
    //

    //c. Напишите программу, реализующую процесс ортогонализации Грама-Шмидта. При этом на хосте создается набор из N векторов 
    // вида a1 = (1,1,1,1,…), a2 = (0,1,1,1,…), a3 = (0,0,1,1,…), … , aN = (…,0,0,0,1). Далее эти вектора передаются на девайс, 
    // после чего производится ортогонализация.---------------------------------------------------------------------------------------------------------------

    ////выделение памяти на хосте для векторов и результатов
    //float* h_vectors = (float*)malloc(N * N * sizeof(float));
    //float* h_ortho_vectors = (float*)malloc(N * N * sizeof(float));

    ////инициализация векторов на хосте
    //for (int i = 0; i < N; ++i) {
    //    for (int j = 0; j < N; ++j) {
    //        if (j >= i) {
    //            h_vectors[i * N + j] = 1.0f;
    //        }
    //        else {
    //            h_vectors[i * N + j] = 0.0f;
    //        }
    //        h_ortho_vectors[i * N + j] = 0.0f; // Инициализация ортогональных векторов
    //    }
    //}

    ////выделение памяти на устройстве для векторов
    //float* d_vectors, * d_ortho_vectors;
    //cudaMalloc((void**)&d_vectors, N * N * sizeof(float));
    //cudaMalloc((void**)&d_ortho_vectors, N * N * sizeof(float));

    ////копирование векторов с хоста на устройство
    //cudaMemcpy(d_vectors, h_vectors, N * N * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_ortho_vectors, h_ortho_vectors, N * N * sizeof(float), cudaMemcpyHostToDevice); // Копирование ортогональных векторов

    ////выполнение ортогонализации Грама-Шмидта
    //gram_schmidt << <numBlocks, threadsPerBlock >> > (d_vectors, d_ortho_vectors, N);

    ////копирование результатов обратно на хост
    //cudaMemcpy(h_vectors, d_vectors, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    ////вывод результатов
    //printf("Orthogonalized vectors:\n");
    //for (int i = 0; i < N; ++i) {
    //    printf("Vector %d: ", i + 1);
    //    for (int j = 0; j < N; ++j) {
    //        printf("%.2f ", h_vectors[i * N + j]);
    //    }
    //    printf("\n");
    //}

    ////освобождение памяти на хосте и устройстве
    //free(h_vectors);
    //free(h_ortho_vectors);
    //cudaFree(d_vectors);
    //cudaFree(d_ortho_vectors);


//d. Проверьте, являются ли две заданные квадратные матрицы A и B коммутирующими (т.е. их произведение не зависит от порядка умножения: AB = BA).--------
    
//задание матриц A и B
    //float A[N][N] = {
    //    {1, 2, 3, 4},
    //    {5, 6, 7, 8},
    //    {9, 10, 11, 12},
    //    {13, 14, 15, 16}
    //};
    //float B[N][N] = {
    //    {1, 2, 3, 4},
    //    {5, 6, 7, 8},
    //    {9, 10, 11, 12},
    //    {13, 14, 15, 16}
    //};

    ////выделение памяти на устройстве для матриц
    //float* d_A, * d_B, * d_C1, * d_C2;
    //cudaMalloc((void**)&d_A, N * N * sizeof(float));
    //cudaMalloc((void**)&d_B, N * N * sizeof(float));
    //cudaMalloc((void**)&d_C1, N * N * sizeof(float));
    //cudaMalloc((void**)&d_C2, N * N * sizeof(float));

    ////копирование матриц с хоста на устройство
    //cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    ////выполнение умножения матриц в обоих порядках
    //dim3 threadsPerBlock(16, 16);
    //dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //matrix_multiply << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C1, N);
    //matrix_multiply << <numBlocks, threadsPerBlock >> > (d_B, d_A, d_C2, N);

    ////копирование результатов обратно на хост
    //float h_C1[N][N], h_C2[N][N];
    //cudaMemcpy(h_C1, d_C1, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_C2, d_C2, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    ////сравнение результатов умножения
    //bool commutative = true;
    //for (int i = 0; i < N; ++i) {
    //    for (int j = 0; j < N; ++j) {
    //        if (h_C1[i][j] != h_C2[i][j]) {
    //            commutative = false;
    //            break;
    //        }
    //    }
    //}

    ////освобождение памяти на устройстве
    //cudaFree(d_A);
    //cudaFree(d_B);
    //cudaFree(d_C1);
    //cudaFree(d_C2);

    ////проверка коммутативности матриц
    //if (commutative) {
    //    printf("Matrices are commutative.\n");
    //}
    //else {
    //    printf("Matrices are not commutative.\n");
    //}


    //e. Напишите функцию для сложения двух матриц M×N, заданных как вектор векторов (float a[M][N], b[M][N]).------------------------------------------ 
    // Матрицы задаются на хосте, после копируются на девайс, и там производится сложение.
    // Задание матриц A и B

    //задание матриц A и B и С
    float A[M][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    float B[M][N] = {
        {12, 11, 10, 9},
        {8, 7, 6, 5},
        {4, 3, 2, 1}
    };
    float C[M][N];

    //выделение памяти на устройстве
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, M * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    //копирование матриц A и B с хоста на устройство
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    //выполнение сложения матриц на устройстве
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_add << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, M, N);

    //копирование результата с устройства на хост
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    //вывод результата
    printf("Result matrix C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    //освобождение памяти на устройстве
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
