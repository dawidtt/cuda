#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
  Sekwencyjna (CPU) wersja obliczania sumy w promieniu R
  TAB - tablica NxN zapisana liniowo: TAB[i*N + j]
  OUT - tablica (N-2R)x(N-2R)
  Dane typu float
*/


// Inicjalizacja danych niejednorodnych

void init_random(float* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (float)rand() / RAND_MAX;
    }
}


// Referencyjna wersja sekwencyjna (1 wątek CPU)

void cpu_stencil(
    const float* TAB,
    float* OUT,
    int N,
    int R
) {
    int outN = N - 2 * R;

    for (int i = R; i < N - R; ++i) {
        for (int j = R; j < N - R; ++j) {

            float sum = 0.0f;

            for (int di = -R; di <= R; ++di) {
                for (int dj = -R; dj <= R; ++dj) {
                    sum += TAB[(i + di) * N + (j + dj)];
                }
            }

            OUT[(i - R) * outN + (j - R)] = sum;
        }
    }
}


int main(int argc, char** argv) {

    int N = 1024;   // rozmiar macierzy wejściowej
    int R = 1;     // promień

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) R = atoi(argv[2]);

    if (N <= 2 * R) {
        printf("Błąd: N musi być > 2*R\n");
        return 1;
    }

    int outN = N - 2 * R;

    float* TAB = (float*)malloc(N * N * sizeof(float));
    float* OUT = (float*)malloc(outN * outN * sizeof(float));

    if (!TAB || !OUT) {
        printf("Błąd alokacji pamięci\n");
        return 1;
    }

   srand(time(NULL));

    int homogeneous_test = 0; // 1 = test jednorodny, 0 = losowe dane

    if (homogeneous_test) {
        // Wypełnienie tablicy wszystkimi 1.0 (jednorodne)
        for (int i = 0; i < N*N; i++)
            TAB[i] = 1.0f;
    } else {
     
        init_random(TAB, N * N);
    }

    // Test czasowy na CPU
    clock_t t0 = clock();
    cpu_stencil(TAB, OUT, N, R);
    clock_t t1 = clock();

    // Test jednorodny po obliczeniach
    if (homogeneous_test) {
        float expected = (float)((2*R+1)*(2*R+1));
        int errors = 0;
        for (int i = 0; i < outN*outN; i++) {
            if (fabs(OUT[i] - expected) > 1e-5)
                errors++;
        }
        if (errors == 0)
            printf("Test jednorodny przeszedł poprawnie!\n");
        else
            printf("Błąd w testach jednorodnych: %d błędów\n", errors);
    }


    double time_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Przykładowe wartości do sanity check
    printf("OUT[0]        = %f\n", OUT[0]);
    printf("OUT[center]   = %f\n",
           OUT[(outN / 2) * outN + (outN / 2)]);
    printf("Czas CPU: %.6f s\n", time_sec);

    // Liczenie FLOP/s (tylko dodawania)
    long long ops_per_elem = (long long)(2 * R + 1) * (2 * R + 1);
    long long total_ops =
        (long long)outN * outN * ops_per_elem;

    double flops = total_ops / time_sec;

    printf("Operacje: %lld\n", total_ops);
    printf("Wydajność CPU: %.3e FLOP/s\n", flops);

    free(TAB);
    free(OUT);

    return 0;
}
