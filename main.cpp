// Headers
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <mkl_vsl.h>

using namespace std;
using namespace std::chrono;

void monte_carlo_option_pricing(float *S0, float *K, float *T, float *v, float *r, float *C, float *z, int numOptions, int M)
{
    for (int opt = 0; opt < 5; opt++)
    {
        float sumPrices = 0.0;
#pragma omp parallel for
        for (int i = 0; i < M; i++)
        {
            float ST = S0[opt] * expf((r[opt] - v[opt] * v[opt] * 0.5) * T[opt] + v[opt] * z[i] * sqrtf(T[opt]));
            sumPrices += max(ST - K[opt], 0.0f);
        }
        C[opt] = expf(-r[opt] * T[opt]) * sumPrices / M;
    }
}

int main()
{
    auto t1 = high_resolution_clock::now();

    static const int M = 5000000;

    float *z = new float[M];
    float *S0 = new float[5];
    float *K = new float[5];
    float *T = new float[5];
    float *v = new float[5];
    float *r = new float[5];
    float *C = new float[5];

#pragma omp parallel for
    for (int i = 0; i < 5; i++)
    {
        S0[i] = 90.0f + 5.0f * i;
        r[i] = 0.01f + 0.01f * i;
        v[i] = 0.3f;
        K[i] = 100.0f;
    }

    T[0] = 1.0f;
    T[1] = 1.2f;
    T[2] = 1.5f;
    T[3] = 2.0f;
    T[4] = 2.5f;

    const int SEED = 42;
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, SEED);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, M, z, 0.0f, 1.0f);

    monte_carlo_option_pricing(S0, K, T, v, r, C, z, 5, M);

    auto t2 = high_resolution_clock::now();

    for (int i = 0; i < 5; i++)
    {
        cout << "Option Price " << i + 1 << "): " << C[i] << '\n';
    }

    cout << "\nTime taken to price the given " << 5 << " options is: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << std::endl;

    vslDeleteStream(&stream);

    return 0;
}
