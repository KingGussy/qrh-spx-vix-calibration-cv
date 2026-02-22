
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

// Kernels and wrappers to (a) loop over the paths for the MC sim and (b) compute the forward VIX deterministically

namespace py = pybind11;

// Helpers:

// ------------------ CUDA error helper ------------------
#define CUDA_CHECK(expr) do {                                    \
    cudaError_t __err = (expr);                                  \
    if (__err != cudaSuccess) {                                  \
        throw std::runtime_error(                                \
            std::string("CUDA error: ") +                        \
            cudaGetErrorString(__err) + " at " + __FILE__ + ":" +\
            std::to_string(__LINE__));                           \
    }                                                            \
} while(0)

// ------------------ RNG init kernel ------------------
__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* states,
                                int m,
                                unsigned long long seed_base)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    // sequence = path id, offset = 0
    curand_init(seed_base, /*sequence*/ static_cast<unsigned long long>(i),
                /*offset*/ 0ULL, &states[i]);
}

__global__ void dot_cZT_kernel(
    const double* __restrict__ ZQ, // (m*n)
    const double* __restrict__ c,  // (n)
    double* __restrict__ out,      // (m)
    int m, int n
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const double* zi = ZQ + (long long)i * n;
    double acc = 0.0;
    for (int j = 0; j < n; ++j) acc += zi[j] * c[j];
    out[i] = acc;
}


// (a) the MC

__global__ void step_kernel_qphi(
    // path state + outputs
    double* __restrict__ ZQ,      // (m*n)
    double* __restrict__ ZL,      // (m*n) or nullptr if !use_cv
    double* __restrict__ I_Q,     // (m)
    double* __restrict__ I_L,     // (m) or nullptr if !use_cv
    double* __restrict__ S_Q,     // (m) or nullptr if !return_ST
    double* __restrict__ S_L,     // (m) or nullptr if !return_ST || !use_cv
    double* __restrict__ J_Q,     // (m) or nullptr
    double* __restrict__ J_L,     // (m) or nullptr

    // constants / coeffs
    const double* __restrict__ c,      // (n)
    const double* __restrict__ decay,  // (n)
    const double* __restrict__ kfac,   // (n)
    const double* __restrict__ inv,    // (n)
    int n,

    // QRH params
    double qa, double qb, double qc0, double qlam, double qeta,
    // LRH params
    double lalpha, double lbeta, double llam, double leta,
    // step config
    double dt, double sqrt_dt,
    int quad_left, int use_kfac, int use_cv, int use_cap, double z_cap,

    // QRH phi-capping
    int qphi_on, double x_star,

    // RNG
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    int m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    // Per path pointers
    double* zqi = ZQ + static_cast<long long>(i) * n;
    double* zli = (use_cv && ZL) ? (ZL + static_cast<long long>(i) * n) : nullptr;

    // Shared brownian stream dW for QRH/LRH on this path and step
    curandStatePhilox4_32_10_t local = rng_states[i];
    double dW = curand_normal_double(&local) * sqrt_dt;
    rng_states[i] = local;

    // Precompute x_star^2 if on
    const double x_star2 = (qphi_on ? (x_star * x_star) : 0.0);

    // ------------------- QRH -------------------
    double Zsum = 0.0;
    for (int j = 0; j < n; ++j) Zsum += zqi[j] * c[j];

    // V_k = qa * phi(Zsum - qb) + qc0
    double xk  = (Zsum - qb);
    double xk2 = xk * xk;
    if (qphi_on && xk2 > x_star2) xk2 = x_star2;

    double V_k = qa * xk2 + qc0;
    if (V_k < 0.0) V_k = 0.0;

    const double driftQ = -qlam * Zsum;
    const double noiseQ = qeta * sqrt(V_k) * dW;

    double Zsum_next = 0.0;
    if (use_kfac) {
        for (int j = 0; j < n; ++j) {
            double znew = decay[j] * zqi[j] + driftQ * kfac[j] + noiseQ;
            if (use_cap) {
                if (znew >  z_cap) znew =  z_cap;
                if (znew < -z_cap) znew = -z_cap;
            }
            zqi[j] = znew;
            Zsum_next += zqi[j] * c[j];
        }
    } else {
        for (int j = 0; j < n; ++j) {
            double znew = (zqi[j] + driftQ * dt + noiseQ) * inv[j];
            if (use_cap) {
                if (znew >  z_cap) znew =  z_cap;
                if (znew < -z_cap) znew = -z_cap;
            }
            zqi[j] = znew;
            Zsum_next += zqi[j] * c[j];
        }
    }

    // V_next = qa * phi(Zsum_next - qb) + qc0
    double xn  = (Zsum_next - qb);
    double xn2 = xn * xn;
    if (qphi_on && xn2 > x_star2) xn2 = x_star2;

    double V_next = qa * xn2 + qc0;
    if (V_next < 0.0) V_next = 0.0;

    const double V_for_step = quad_left ? V_k : V_next;
    I_Q[i] += V_for_step * dt;

    const double Z_for_step = quad_left ? Zsum : Zsum_next;
    if (J_Q) J_Q[i] += Z_for_step * dt;

    if (S_Q) {
        double incrQ = sqrt(V_for_step) * dW - 0.5 * V_for_step * dt;
        S_Q[i] *= exp(incrQ);
    }

    // Control variate side- no variance cap to keep fourier-consistent
    // note the z_cap here is not used, capping is only done in the QRH side via the x_cap, this is vestigial from an old iteration
    if (use_cv && zli) {
        double ZsumL = 0.0;
        for (int j = 0; j < n; ++j) ZsumL += zli[j] * c[j];

        double VL_k = lalpha + lbeta * ZsumL;
        if (VL_k < 0.0) VL_k = 0.0;

        const double driftL = -llam * ZsumL;
        const double noiseL = leta * sqrt(VL_k) * dW;

        double ZsumL_next = 0.0;
        if (use_kfac) {
            for (int j = 0; j < n; ++j) {
                double znew = decay[j] * zli[j] + driftL * kfac[j] + noiseL;
                if (use_cap) {
                    if (znew >  z_cap) znew =  z_cap;
                    if (znew < -z_cap) znew = -z_cap;
                }
                zli[j] = znew;
                ZsumL_next += zli[j] * c[j];
            }
        } else {
            for (int j = 0; j < n; ++j) {
                double znew = (zli[j] + driftL * dt + noiseL) * inv[j];
                if (use_cap) {
                    if (znew >  z_cap) znew =  z_cap;
                    if (znew < -z_cap) znew = -z_cap;
                }
                zli[j] = znew;
                ZsumL_next += zli[j] * c[j];
            }
        }

        double VL_next = lalpha + lbeta * ZsumL_next;
        if (VL_next < 0.0) VL_next = 0.0;

        const double V_for_step_L = quad_left ? VL_k : VL_next;
        I_L[i] += V_for_step_L * dt;

        const double ZL_for_step = quad_left ? ZsumL : ZsumL_next;
        if (J_L) J_L[i] += ZL_for_step * dt;

        if (S_L) {
            double incrL = sqrt(V_for_step_L) * dW - 0.5 * V_for_step_L * dt;
            S_L[i] *= exp(incrL);
        }
    }
}
// THIS GUY
py::tuple simulate_paths_cuda(
    int m,
    py::object kernel,   // c, gamma
    py::object q,        // a, b, c, eta, lambda, z0
    py::object l,        // alpha, beta, eta, lambda
    double T,
    int n_steps,
    bool use_CV               = true,
    const std::string& scheme = "inv",    // inv or kfac
    const std::string& quad   = "left",   // left or right (always left on left for consistency)
    double S0                 = 100.0,
    bool return_ST            = false,

    // Outputs
    bool return_ZT            = false,
    bool return_cZT           = false,
    bool return_J             = false,

    // pass float x_star to enable QRH phi-cap: phi(x)=min(x^2, x_star^2) with x=Zsum-qb
    py::object vcap_obj       = py::none(),

    double v_floor            = 0.0,          // unused (still clamp >=0 in kernel)
    py::object z_cap_obj      = py::none(),
    py::object seed_obj       = py::none(),
    py::object dW_shared_obj  = py::none()
) {
    if (!dW_shared_obj.is_none()) {
        throw std::runtime_error("Requires dW_shared=None (uses cuRAND).");
    }

    // ---- Pull kernel arrays ----
    py::array_t<double> c_arr     = kernel.attr("c").cast<py::array_t<double>>();
    py::array_t<double> gamma_arr = kernel.attr("gamma").cast<py::array_t<double>>();
    auto c_info = c_arr.request(); auto g_info = gamma_arr.request();
    if (c_info.ndim != 1 || g_info.ndim != 1) throw std::runtime_error("kernel.c/gamma must be 1D");
    if (g_info.shape[0] != c_info.shape[0])   throw std::runtime_error("c and gamma length mismatch");
    const int n = static_cast<int>(c_info.shape[0]);
    const double* c_host     = static_cast<const double*>(c_info.ptr);
    const double* gamma_host = static_cast<const double*>(g_info.ptr);

    // ---- Params ----
    const double qa   = q.attr("a").cast<double>();
    const double qb   = q.attr("b").cast<double>();
    const double qc0  = q.attr("c0").cast<double>();
    const double qlam = q.attr("lam").cast<double>();
    const double qeta = q.attr("eta").cast<double>();

    py::array_t<double> z0_arr = q.attr("z0").cast<py::array_t<double>>();
    auto z0_info = z0_arr.request();
    if (z0_info.ndim != 1 || static_cast<int>(z0_info.shape[0]) != n)
        throw std::runtime_error("q.z0 must be 1-D of length n");
    const double* z0_host = static_cast<const double*>(z0_info.ptr);

    double lalpha=0.0, lbeta=0.0, llam=0.0, leta=0.0;
    const int use_cv = use_CV ? 1 : 0;
    if (use_cv) {
        lalpha = l.attr("alpha").cast<double>();
        lbeta  = l.attr("beta").cast<double>();
        llam   = l.attr("lam").cast<double>();
        leta   = l.attr("eta").cast<double>();
    }

    // ---- Grid / coeffs ----
    const int    N       = std::max(1, n_steps);
    const double dt      = T / double(N);
    const double sqrt_dt = std::sqrt(dt);
    const int    use_kfac  = (scheme == "kfac") ? 1 : 0;
    const int    quad_left = (quad   == "left") ? 1 : 0;
    const int    use_cap   = !z_cap_obj.is_none() ? 1 : 0;
    const double z_cap     = use_cap ? z_cap_obj.cast<double>() : 0.0;

    // ---- QRH phi-cap controls ----
    const int    qphi_on = (!vcap_obj.is_none()) ? 1 : 0;
    const double x_star  = (!vcap_obj.is_none()) ? vcap_obj.cast<double>() : 0.0;

    // Precompute decay/kfac/inv on host
    std::vector<double> decay_h(n), kfac_h(n), inv_h(n);
    for (int j = 0; j < n; ++j) {
        const double g = gamma_host[j];
        const double d = std::exp(-g * dt);
        decay_h[j] = d;
        kfac_h[j]  = (g > 1e-12) ? (1.0 - d) / g : dt;
        inv_h[j]   = 1.0 / (1.0 + g * dt);
    }

    // ---- manage device memory ----
    double *d_c=nullptr, *d_decay=nullptr, *d_kfac=nullptr, *d_inv=nullptr;
    double *d_ZQ=nullptr, *d_ZL=nullptr;
    double *d_IQ=nullptr, *d_IL=nullptr;
    double *d_SQ=nullptr, *d_SL=nullptr;
    double *d_JQ=nullptr, *d_JL=nullptr;
    curandStatePhilox4_32_10_t* d_states=nullptr;

    CUDA_CHECK(cudaMalloc(&d_c,     sizeof(double)*n));
    CUDA_CHECK(cudaMalloc(&d_decay, sizeof(double)*n));
    CUDA_CHECK(cudaMalloc(&d_kfac,  sizeof(double)*n));
    CUDA_CHECK(cudaMalloc(&d_inv,   sizeof(double)*n));
    CUDA_CHECK(cudaMemcpy(d_c,     c_host,         sizeof(double)*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_decay, decay_h.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kfac,  kfac_h.data(),  sizeof(double)*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv,   inv_h.data(),   sizeof(double)*n, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_ZQ, sizeof(double)* (long long)m * n));
    {
        std::vector<double> Z0((long long)m*n);
        for (int i = 0; i < m; ++i) {
            std::copy(z0_host, z0_host + n, Z0.data() + (long long)i*n);
        }
        CUDA_CHECK(cudaMemcpy(d_ZQ, Z0.data(), sizeof(double)* (long long)m * n, cudaMemcpyHostToDevice));
    }

    if (use_cv) {
        CUDA_CHECK(cudaMalloc(&d_ZL, sizeof(double)* (long long)m * n));
        CUDA_CHECK(cudaMemset(d_ZL, 0, sizeof(double)* (long long)m * n));
    }

    CUDA_CHECK(cudaMalloc(&d_IQ, sizeof(double)*m));
    CUDA_CHECK(cudaMemset(d_IQ, 0, sizeof(double)*m));
    if (use_cv) {
        CUDA_CHECK(cudaMalloc(&d_IL, sizeof(double)*m));
        CUDA_CHECK(cudaMemset(d_IL, 0, sizeof(double)*m));
    }

    if (return_J) {
        CUDA_CHECK(cudaMalloc(&d_JQ, sizeof(double)*m));
        CUDA_CHECK(cudaMemset(d_JQ, 0, sizeof(double)*m));
        if (use_cv) {
            CUDA_CHECK(cudaMalloc(&d_JL, sizeof(double)*m));
            CUDA_CHECK(cudaMemset(d_JL, 0, sizeof(double)*m));
        }
    }

    if (return_ST) {
        CUDA_CHECK(cudaMalloc(&d_SQ, sizeof(double)*m));
        if (use_cv) CUDA_CHECK(cudaMalloc(&d_SL, sizeof(double)*m));
        std::vector<double> S0vec(m, S0);
        CUDA_CHECK(cudaMemcpy(d_SQ, S0vec.data(), sizeof(double)*m, cudaMemcpyHostToDevice));
        if (use_cv) CUDA_CHECK(cudaMemcpy(d_SL, S0vec.data(), sizeof(double)*m, cudaMemcpyHostToDevice));
    }

    // RNG states
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandStatePhilox4_32_10_t)*m));
    unsigned long long seed_base =
        seed_obj.is_none() ? 0xA5A5A5A55A5A5A5AULL
                           : static_cast<unsigned long long>(seed_obj.cast<std::uint64_t>());

    int block = 256;
    int grid  = (m + block - 1) / block;

    init_rng_kernel<<<grid, block>>>(d_states, m, seed_base);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // time step loop kernel
    for (int k = 0; k < N; ++k) {
        step_kernel_qphi<<<grid, block>>>(
            d_ZQ, d_ZL,
            d_IQ, d_IL,
            d_SQ, d_SL,
            (return_J ? d_JQ : nullptr),
            (return_J && use_cv ? d_JL : nullptr),
            d_c, d_decay, d_kfac, d_inv, n,
            qa,qb,qc0, qlam,qeta,
            lalpha,lbeta,llam,leta,
            dt, sqrt_dt,
            (quad_left?1:0),
            (use_kfac?1:0),
            use_cv,
            (use_cap?1:0), z_cap,
            (qphi_on?1:0), x_star,
            d_states, m
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy main outputs ----
    py::array_t<double> I_Q_py({m});
    {
        auto b = I_Q_py.request();
        CUDA_CHECK(cudaMemcpy(b.ptr, d_IQ, sizeof(double)*m, cudaMemcpyDeviceToHost));
    }

    py::object I_L_out = py::none();
    if (use_cv) {
        py::array_t<double> I_L_py({m});
        auto b = I_L_py.request();
        CUDA_CHECK(cudaMemcpy(b.ptr, d_IL, sizeof(double)*m, cudaMemcpyDeviceToHost));
        I_L_out = py::object(I_L_py);
    }

    py::object S_tuple = py::make_tuple(py::none(), py::none());
    if (return_ST) {
        py::array_t<double> S_Q_py({m});
        auto bq = S_Q_py.request();
        CUDA_CHECK(cudaMemcpy(bq.ptr, d_SQ, sizeof(double)*m, cudaMemcpyDeviceToHost));

        py::object S_L_obj = py::none();
        if (use_cv) {
            py::array_t<double> S_L_py({m});
            auto bl = S_L_py.request();
            CUDA_CHECK(cudaMemcpy(bl.ptr, d_SL, sizeof(double)*m, cudaMemcpyDeviceToHost));
            S_L_obj = py::object(S_L_py);
        }
        S_tuple = py::make_tuple(py::object(S_Q_py), S_L_obj);
    }

    // ---- terminal Z_T and c^T Z_T ----
    py::object ZT_out  = py::none();
    py::object cZT_out = py::none();

    if (return_ZT || return_cZT) {
        if (return_ZT) {
            py::array_t<double> Z_T_py({m, n});
            auto bz = Z_T_py.request();
            CUDA_CHECK(cudaMemcpy(bz.ptr, d_ZQ, sizeof(double)*(size_t)m*n, cudaMemcpyDeviceToHost));
            ZT_out = py::object(Z_T_py);
        }
        if (return_cZT) {
            double* d_cZT = nullptr;
            CUDA_CHECK(cudaMalloc(&d_cZT, sizeof(double)*m));
            dot_cZT_kernel<<<grid, block>>>(d_ZQ, d_c, d_cZT, m, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            py::array_t<double> cZT_py({m});
            auto bc = cZT_py.request();
            CUDA_CHECK(cudaMemcpy(bc.ptr, d_cZT, sizeof(double)*m, cudaMemcpyDeviceToHost));
            cZT_out = py::object(cZT_py);

            cudaFree(d_cZT);
        }
    }

    py::object J_Q_out = py::none();
    py::object J_L_out = py::none();
    if (return_J) {
        py::array_t<double> J_Q_py({m});
        auto bjq = J_Q_py.request();
        CUDA_CHECK(cudaMemcpy(bjq.ptr, d_JQ, sizeof(double)*m, cudaMemcpyDeviceToHost));
        J_Q_out = py::object(J_Q_py);

        if (use_cv) {
            py::array_t<double> J_L_py({m});
            auto bjl = J_L_py.request();
            CUDA_CHECK(cudaMemcpy(bjl.ptr, d_JL, sizeof(double)*m, cudaMemcpyDeviceToHost));
            J_L_out = py::object(J_L_py);
        }
    }

    // ---- cleanup ----
    cudaFree(d_c); cudaFree(d_decay); cudaFree(d_kfac); cudaFree(d_inv);
    cudaFree(d_ZQ); if (d_ZL) cudaFree(d_ZL);
    cudaFree(d_IQ); if (d_IL) cudaFree(d_IL);
    if (d_JQ) cudaFree(d_JQ); if (d_JL) cudaFree(d_JL);
    if (d_SQ) cudaFree(d_SQ); if (d_SL) cudaFree(d_SL);
    cudaFree(d_states);

    if (!return_J) {
        return py::make_tuple(py::object(I_Q_py), I_L_out, S_tuple, ZT_out, cZT_out);
    } else {
        return py::make_tuple(py::object(I_Q_py), I_L_out, S_tuple, ZT_out, cZT_out, J_Q_out, J_L_out);
    }
}



// ======================================================================================================
// (b) The deterministic VIX stuff
// forward xi per path
__global__ void xi_forward_kernel(
    // per-path inputs
    const double* __restrict__ ZT,       // (m * n)
    int m, int n, int n_steps,

    // QRH params
    double a, double b, double c0, double lam, double eta,

    // kernel / scheme
    const double* __restrict__ c,        // (n)
    const double* __restrict__ gamma,    // (n)
    double delta,

    // rank-1 kernel precompute
    const double* __restrict__ Vc_sq,    // (n_steps+1), Vc_sq[r] = (c^T v_r)^2
    double denom,                        // 1 - a*eta^2*delta*Vc_sq[0]

    // output
    double* __restrict__ XI              // (m * n_steps)
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= m) return;

    const double* zt = ZT + (size_t)pid * n;
    double* xi = XI + (size_t)pid * n_steps;

    // working state m_j (length n), initialise with m_0 = Z_T(path)
    extern __shared__ double sh[];
    double* m_j = sh + (size_t)threadIdx.x * (size_t)n; // per-thread slice
    for (int i = 0; i < n; ++i) m_j[i] = zt[i];

    // Precompute diagonal step (inv scheme)
    // D_i = 1 - gamma_i * delta
    // We'll reuse it in-place to update m_j each j-step.
    // Nothing to store globally.
    for (int j = 0; j < n_steps; ++j) {
        // 1) mean step: m_{j} -> m_{j} (in-place) to represent m_{j} at this grid index
        //    Apply diagonal part, then rank-1 coupling: -lam*delta*(c^T m)*u
        // Diagonal OU:
        // double dot_cm = 0.0;
        // for (int i = 0; i < n; ++i) {
        //     m_j[i] = m_j[i] * (1.0 - gamma[i] * delta);
        //     dot_cm += c[i] * m_j[i];
        // }
        // // rank-1 coupling:
        // double r1 = lam * delta * dot_cm;  // scalar
        // for (int i = 0; i < n; ++i) m_j[i] -= r1;

        // IMEX mean step: (I + delta*Gamma) m_new = m_old - lam*delta*(c^T m_old) * 1
        double cv_old = 0.0;
        for (int i = 0; i < n; ++i) cv_old += c[i] * m_j[i];   // dot from OLD m

        double r1 = lam * delta * cv_old;                      // scalar
        for (int i = 0; i < n; ++i) {
            double denom_i = 1.0 + gamma[i] * delta;           // implicit diagonal
            m_j[i] = (m_j[i] - r1) / denom_i;
        }

        // 2) B_j
        double x = 0.0;
        for (int i = 0; i < n; ++i) x += c[i] * m_j[i];
        double Bj = a * (x - b) * (x - b) + c0;
        if (Bj < 0.0) Bj = 0.0;

        // 3) accum = sum_{r=1..j} xi_{j-r} * Vc_sq[r]    (since k<j ↔ r=j-k)
        double accum = 0.0;
        for (int r = 1; r <= j; ++r) {
            accum += xi[j - r] * Vc_sq[r];
        }

        // 4) xi_j
        double numer = Bj + (a * eta * eta) * delta * accum;
        double val = numer / denom;
        if (val < 0.0) val = 0.0;
        xi[j] = val;
    }
}

// precompute denom on CPU
static void host_build_Vc_sq(
    std::vector<double>& Vc_sq, // out: size n_steps+1
    double& denom,              // out
    const double* c, const double* gamma, int n,
    double lam, double delta, int n_steps,
    double a, double eta
) {
    // v_0 = u (ones)
    std::vector<double> v(n, 1.0);
    Vc_sq.resize(n_steps + 1);

    auto dot = [&](const std::vector<double>& u, const double* w)->double{
        double s=0.0; for (int i=0;i<n;++i) s += u[i]*w[i]; return s;
    };
    double Vc0 = dot(v, c);
    Vc_sq[0] = Vc0 * Vc0;

    // Step: v <- (I - ΔΓ) v  -  Δλ (c^T v) u
    for (int j = 1; j <= n_steps; ++j) {
        // diagonal OU
        // double cv = dot(v, c);
        // for (int i=0;i<n;++i) v[i] = v[i] * (1.0 - gamma[i]*delta) - lam*delta*cv;
        // double Vcj = dot(v, c);
        // Vc_sq[j] = Vcj * Vcj;
        
        // IMEX: diagonal implicit, rank-1 explicit
        double cv_old = dot(v, c);
        double r1 = lam * delta * cv_old;

        for (int i = 0; i < n; ++i) {
            double denom_i = 1.0 + gamma[i] * delta;
            v[i] = (v[i] - r1) / denom_i;
        }

        double Vcj = dot(v, c);
        Vc_sq[j] = Vcj * Vcj;
    }

    // denom = 1 - a*eta^2*Δ*Vc_sq[0]
    double d = 1.0 - a * eta * eta * delta * Vc_sq[0];
    if (d < 1e-12) d = 1e-12;
    denom = d;
}

// wrapper
py::array_t<double> solve_xi_cuda(
    py::array_t<double> ZT,    // (m, n)
    double a, double b, double c0, double lam, double eta,
    py::array_t<double> c_arr,     // (n,)
    py::array_t<double> gamma_arr, // (n,)
    double delta,
    int n_steps
) {
    auto ZT_i = ZT.request();
    auto c_i  = c_arr.request();
    auto g_i  = gamma_arr.request();
    if (ZT_i.ndim != 2) throw std::runtime_error("ZT must be 2-D (m,n)");
    if (c_i.ndim  != 1 || g_i.ndim != 1) throw std::runtime_error("c,gamma must be 1-D");

    const int m = (int)ZT_i.shape[0];
    const int n = (int)ZT_i.shape[1];
    if ((int)c_i.shape[0] != n || (int)g_i.shape[0] != n)
        throw std::runtime_error("c,gamma length must match ZT.shape[1]");

    const double* ZT_h = static_cast<const double*>(ZT_i.ptr);
    const double* c_h  = static_cast<const double*>(c_i.ptr);
    const double* g_h  = static_cast<const double*>(g_i.ptr);

    // --- host precompute of Vc_sq and denom ---
    std::vector<double> Vc_sq_h;
    double denom = 1.0;
    host_build_Vc_sq(Vc_sq_h, denom, c_h, g_h, n, lam, delta, n_steps, a, eta);

    // --- device buffers ---
    double *d_ZT=nullptr, *d_c=nullptr, *d_g=nullptr, *d_Vc_sq=nullptr, *d_XI=nullptr;
    CUDA_CHECK(cudaMalloc(&d_ZT,    sizeof(double)* (size_t)m*n));
    CUDA_CHECK(cudaMalloc(&d_c,     sizeof(double)* n));
    CUDA_CHECK(cudaMalloc(&d_g,     sizeof(double)* n));
    CUDA_CHECK(cudaMalloc(&d_Vc_sq, sizeof(double)* (size_t)(n_steps+1)));
    CUDA_CHECK(cudaMalloc(&d_XI,    sizeof(double)* (size_t)m*n_steps));

    CUDA_CHECK(cudaMemcpy(d_ZT, ZT_h, sizeof(double)* (size_t)m*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c,  c_h,  sizeof(double)* n,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g,  g_h,  sizeof(double)* n,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Vc_sq, Vc_sq_h.data(), sizeof(double)*(size_t)(n_steps+1), cudaMemcpyHostToDevice));

    // --- launch ---
    int block = 128;
    int grid  = (m + block - 1) / block;
    size_t shmem = (size_t)block * (size_t)n * sizeof(double); // per-thread m_j

    xi_forward_kernel<<<grid, block, shmem>>>(
        d_ZT, m, n, n_steps,
        a, b, c0, lam, eta,
        d_c, d_g, delta,
        d_Vc_sq, denom,
        d_XI
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- copy back ---
    py::array_t<double> XI_py({m, n_steps});
    auto xinfo = XI_py.request();
    CUDA_CHECK(cudaMemcpy(xinfo.ptr, d_XI, sizeof(double)*(size_t)m*n_steps, cudaMemcpyDeviceToHost));

    // --- cleanup ---
    cudaFree(d_ZT); cudaFree(d_c); cudaFree(d_g); cudaFree(d_Vc_sq); cudaFree(d_XI);

    return XI_py;
}

// ------------------ Pybind module (separate name) ------------------
PYBIND11_MODULE(qrh_sim_cuda, m) {
    m.doc() = "CUDA-accelerated QRH/LRH simulator (per-path threads, per-step launches)";
    m.def("simulate_paths_cuda", &simulate_paths_cuda,
          py::arg("m"),
          py::arg("kernel"),
          py::arg("q"),
          py::arg("l"),
          py::arg("T"),
          py::arg("n_steps"),
          py::arg("use_CV")    = true,
          py::arg("scheme")    = "inv",
          py::arg("quad")      = "left",
          py::arg("S0")        = 100.0,
          py::arg("return_ST") = false,
          py::arg("return_ZT") = false,
          py::arg("return_cZT")= false,
          py::arg("return_J")  = false,
          py::arg("vcap_obj")  = py::none(),
          py::arg("v_floor")   = 0.0,
          py::arg("z_cap_obj")     = py::none(),
          py::arg("seed_obj")      = py::none(),
          py::arg("dW_shared_obj") = py::none());

    m.def("solve_xi_cuda", &solve_xi_cuda,
        py::arg("ZT"),
        py::arg("a"), py::arg("b"), py::arg("c0"),
        py::arg("lam"), py::arg("eta"),
        py::arg("c"), py::arg("gamma"),
        py::arg("delta"),
        py::arg("n_steps"),
        "Deterministic forward-variance march (ξ) on GPU; returns (m, n_steps) array.");
}
