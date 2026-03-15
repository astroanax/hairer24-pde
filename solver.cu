// GPU 2D SNS + Lyapunov exponent (Hairer et al. 2024)
// C2C FFT, CUDA, Benettin renormalization
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <random>

#define CK(x) do{cudaError_t e=x;if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CF(x) do{cufftResult r=x;if(r!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT %s:%d %d\n",__FILE__,__LINE__,r);exit(1);}}while(0)

struct P{int Nx=64,Ny=64;double Lx=2*M_PI,Ly=2*M_PI,nu=.005,kappa=.005,sig=2.0,npow=2.0;int kmn=1,kmx=8;double dt=.0005,Tw=5.0,Tr=25.0;int si=200,lr=100;};

__constant__ int cNx,cNy;
__constant__ double cdt,cnu,ckap,csig,cnpow,ckmink,ckmaxk;

__global__ void k_rng(curandState* s,unsigned long sd){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    curand_init(sd,i,0,&s[i]);
}

__global__ void k_stream(const cufftDoubleComplex* w,cufftDoubleComplex* p,const double* k2){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    double inv=k2[i]>1e-12?-1.0/k2[i]:0;
    p[i].x=w[i].x*inv;
    p[i].y=w[i].y*inv;
}

__global__ void k_vel(const cufftDoubleComplex* p,cufftDoubleComplex* u1,cufftDoubleComplex* u2,const double* kx,const double* ky){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ix>=cNx||iy>=cNy)return;
    int idx=ix*cNy+iy;
    u1[idx].x=-ky[iy]*p[idx].y;
    u1[idx].y=ky[iy]*p[idx].x;
    u2[idx].x=kx[ix]*p[idx].y;
    u2[idx].y=-kx[ix]*p[idx].x;
}

__global__ void k_grad(const cufftDoubleComplex* f,cufftDoubleComplex* dfx,cufftDoubleComplex* dfy,const double* kx,const double* ky,const double* dm){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ix>=cNx||iy>=cNy)return;
    int idx=ix*cNy+iy;
    double d=dm[idx];
    dfx[idx].x=-kx[ix]*f[idx].y*d;
    dfx[idx].y=kx[ix]*f[idx].x*d;
    dfy[idx].x=-ky[iy]*f[idx].y*d;
    dfy[idx].y=ky[iy]*f[idx].x*d;
}

__global__ void k_nl(const double* ux,const double* uy,const double* dfx,const double* dfy,double* nl){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    nl[i]=ux[i]*dfx[i]+uy[i]*dfy[i];
}

__global__ void k_deal(cufftDoubleComplex* f,const double* m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    f[i].x*=m[i];
    f[i].y*=m[i];
}

__global__ void k_impl(cufftDoubleComplex* f,const cufftDoubleComplex* rhs,const double* k2,double kap){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    double den=1.0/(1.0+kap*cdt*k2[i]);
    f[i].x=(f[i].x-cdt*rhs[i].x)*den;
    f[i].y=(f[i].y-cdt*rhs[i].y)*den;
}

__global__ void k_noise(cufftDoubleComplex* f,curandState* rng,const double* k2){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    double k=sqrt(k2[i]);
    if(k>=ckmink&&k<=ckmaxk){
        double a=csig/pow(k,cnpow)*sqrt(cdt);
        curandState s=rng[i];
        f[i].x+=a*curand_normal_double(&s);
        f[i].y+=a*curand_normal_double(&s);
        rng[i]=s;
    }
}

__global__ void k_zm(cufftDoubleComplex* f){
    f[0].x=0;
    f[0].y=0;
}

__global__ void k_r2c(const double* r,cufftDoubleComplex* c){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    c[i].x=r[i];
    c[i].y=0;
}

__global__ void k_c2r(const cufftDoubleComplex* c,double* r){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    r[i]=c[i].x;
}

__global__ void k_norm(double* f,double v){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=cNx*cNy)return;
    f[i]*=v;
}

__global__ void k_ss(const double* f,double* p){
    extern __shared__ double s[];
    int t=threadIdx.x;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    s[t]=(i<cNx*cNy)?f[i]*f[i]:0;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j>>=1){
        if(t<j)s[t]+=s[t+j];
        __syncthreads();
    }
    if(t==0)p[blockIdx.x]=s[0];
}

class S {
public:
    P p;
    int N;
    cufftDoubleComplex *dwh,*dph,*drh,*dth,*du1,*du2,*dd1,*dd2,*dnl;
    double *do_,*dr,*dtg,*du1r,*du2r,*dn1,*dd1r,*dd2r;
    double *dkx,*dky,*dk2,*ddm;
    curandState *drng;
    cufftHandle plan;
    dim3 B1,G1,B2,G2;
    std::string od;
    double* d_scratch;
    cufftDoubleComplex *d_tmp1,*d_tmp2,*d_tmp3,*d_tmp4,*d_tmp5;

    S(const P& q, const std::string& o) : p(q), od(o) {
        N = p.Nx * p.Ny;
        B1 = dim3(256);
        G1 = dim3((N + 255) / 256);
        B2 = dim3(16, 16);
        G2 = dim3((p.Nx + 15) / 16, (p.Ny + 15) / 16);

        CK(cudaMalloc(&dwh, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&dph, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&drh, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&dth, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&du1, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&du2, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&dd1, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&dd2, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&dnl, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&do_, N*sizeof(double)));
        CK(cudaMalloc(&dr, N*sizeof(double)));
        CK(cudaMalloc(&dtg, N*sizeof(double)));
        CK(cudaMalloc(&du1r, N*sizeof(double)));
        CK(cudaMalloc(&du2r, N*sizeof(double)));
        CK(cudaMalloc(&dn1, N*sizeof(double)));
        CK(cudaMalloc(&dd1r, N*sizeof(double)));
        CK(cudaMalloc(&dd2r, N*sizeof(double)));
        CK(cudaMalloc(&dkx, p.Nx*sizeof(double)));
        CK(cudaMalloc(&dky, p.Ny*sizeof(double)));
        CK(cudaMalloc(&dk2, N*sizeof(double)));
        CK(cudaMalloc(&ddm, N*sizeof(double)));
        CK(cudaMalloc(&drng, N*sizeof(curandState)));
        CK(cudaMalloc(&d_scratch, G1.x*sizeof(double)));
        // tmp buffers so save() doesn't clobber Fourier data
        CK(cudaMalloc(&d_tmp1, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&d_tmp2, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&d_tmp3, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&d_tmp4, N*sizeof(cufftDoubleComplex)));
        CK(cudaMalloc(&d_tmp5, N*sizeof(cufftDoubleComplex)));

        CK(cudaMemset(dwh, 0, N*sizeof(cufftDoubleComplex)));
        CK(cudaMemset(drh, 0, N*sizeof(cufftDoubleComplex)));
        CK(cudaMemset(dth, 0, N*sizeof(cufftDoubleComplex)));

        init_wk();
        CF(cufftPlan2d(&plan, p.Nx, p.Ny, CUFFT_Z2Z));

        int hNx=p.Nx, hNy=p.Ny;
        CK(cudaMemcpyToSymbol(cNx,&hNx,sizeof(int)));
        CK(cudaMemcpyToSymbol(cNy,&hNy,sizeof(int)));
        double hdt=p.dt,hnu=p.nu,hkap=p.kappa,hsig=p.sig,hnpow=p.npow;
        double hkmn=(double)p.kmn,hkmx=(double)p.kmx;
        CK(cudaMemcpyToSymbol(cdt,&hdt,sizeof(double)));
        CK(cudaMemcpyToSymbol(cnu,&hnu,sizeof(double)));
        CK(cudaMemcpyToSymbol(ckap,&hkap,sizeof(double)));
        CK(cudaMemcpyToSymbol(csig,&hsig,sizeof(double)));
        CK(cudaMemcpyToSymbol(cnpow,&hnpow,sizeof(double)));
        CK(cudaMemcpyToSymbol(ckmink,&hkmn,sizeof(double)));
        CK(cudaMemcpyToSymbol(ckmaxk,&hkmx,sizeof(double)));
        k_rng<<<G1,B1>>>(drng,12345UL);
    }

    ~S() {
        cufftDestroy(plan);
        cudaFree(dwh);cudaFree(dph);cudaFree(drh);cudaFree(dth);
        cudaFree(du1);cudaFree(du2);cudaFree(dd1);cudaFree(dd2);cudaFree(dnl);
        cudaFree(do_);cudaFree(dr);cudaFree(dtg);cudaFree(du1r);cudaFree(du2r);
        cudaFree(dn1);cudaFree(dd1r);cudaFree(dd2r);
        cudaFree(dkx);cudaFree(dky);cudaFree(dk2);cudaFree(ddm);cudaFree(drng);
        cudaFree(d_scratch);
        cudaFree(d_tmp1);cudaFree(d_tmp2);cudaFree(d_tmp3);cudaFree(d_tmp4);cudaFree(d_tmp5);
    }

    void init_wk() {
        std::vector<double> kx(p.Nx),ky(p.Ny),k2(N),dm(N);
        int nxh=p.Nx/2,nyh=p.Ny/2;
        for(int i=0;i<p.Nx;i++){kx[i]=(i<=nxh)?i:i-p.Nx;kx[i]*=2*M_PI/p.Lx;}
        for(int j=0;j<p.Ny;j++){ky[j]=(j<=nyh)?j:j-p.Ny;ky[j]*=2*M_PI/p.Ly;}
        double kc1=(2.0/3.0)*nxh*2*M_PI/p.Lx;
        double kc2=(2.0/3.0)*nyh*2*M_PI/p.Ly;
        for(int i=0;i<p.Nx;i++)
            for(int j=0;j<p.Ny;j++){
                int idx=i*p.Ny+j;
                k2[idx]=kx[i]*kx[i]+ky[j]*ky[j];
                dm[idx]=(fabs(kx[i])<=kc1&&fabs(ky[j])<=kc2)?1.0:0.0;
            }
        CK(cudaMemcpy(dkx,kx.data(),p.Nx*sizeof(double),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dky,ky.data(),p.Ny*sizeof(double),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk2,k2.data(),N*sizeof(double),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ddm,dm.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    }

    void r2c(double* real, cufftDoubleComplex* cmplx) {
        k_r2c<<<G1,B1>>>(real,cmplx);
        CF(cufftExecZ2Z(plan,cmplx,cmplx,CUFFT_FORWARD));
    }

    void c2r(cufftDoubleComplex* cmplx, double* real) {
        CF(cufftExecZ2Z(plan,cmplx,cmplx,CUFFT_INVERSE));
        k_c2r<<<G1,B1>>>(cmplx,real);
        k_norm<<<G1,B1>>>(real,1.0/(double)N);
    }

    void comp_vel() {
        k_stream<<<G1,B1>>>(dwh,dph,dk2);
        k_vel<<<G2,B2>>>(dph,du1,du2,dkx,dky);
        c2r(du1,du1r);
        c2r(du2,du2r);
    }

    void step_vort() {
        comp_vel();
        k_grad<<<G2,B2>>>(dwh,dd1,dd2,dkx,dky,ddm);
        c2r(dd1,dd1r);c2r(dd2,dd2r);
        k_nl<<<G1,B1>>>(du1r,du2r,dd1r,dd2r,dn1);
        r2c(dn1,dnl);
        k_deal<<<G1,B1>>>(dnl,ddm);
        k_impl<<<G1,B1>>>(dwh,dnl,dk2,p.nu);
        k_noise<<<G1,B1>>>(dwh,drng,dk2);
        k_zm<<<1,1>>>(dwh);
    }

    void step_scl(cufftDoubleComplex* sh, double* sr) {
        k_grad<<<G2,B2>>>(sh,dd1,dd2,dkx,dky,ddm);
        c2r(dd1,dd1r);c2r(dd2,dd2r);
        k_nl<<<G1,B1>>>(du1r,du2r,dd1r,dd2r,dn1);
        r2c(dn1,dnl);
        k_deal<<<G1,B1>>>(dnl,ddm);
        k_impl<<<G1,B1>>>(sh,dnl,dk2,p.kappa);
        k_zm<<<1,1>>>(sh);
    }

    double cnorm(double* f) {
        k_ss<<<G1,B1,256*sizeof(double)>>>(f,d_scratch);
        std::vector<double> h(G1.x);
        CK(cudaMemcpy(h.data(),d_scratch,G1.x*sizeof(double),cudaMemcpyDeviceToHost));
        double s=0;
        for(auto& v:h)s+=v;
        return sqrt(s/N);
    }

    void init() {
        std::mt19937 rng(42);
        std::normal_distribution<double> nd(0,1);
        std::vector<cufftDoubleComplex> ini(N);
        for(int i=0;i<p.Nx;i++)
            for(int j=0;j<p.Ny;j++){
                int idx=i*p.Ny+j;
                int kx2=(i<=p.Nx/2)?i:i-p.Nx;
                int ky2=(j<=p.Ny/2)?j:j-p.Ny;
                double k=sqrt((double)(kx2*kx2+ky2*ky2));
                if(k>0&&k<=8){double a=1.0/(1+k*k);ini[idx].x=a*nd(rng);ini[idx].y=a*nd(rng);}
                else{ini[idx].x=0;ini[idx].y=0;}
            }
        ini[0].x=0;ini[0].y=0;
        CK(cudaMemcpy(dwh,ini.data(),N*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));
        for(auto& v:ini){v.x*=0.5;v.y*=0.5;}
        CK(cudaMemcpy(drh,ini.data(),N*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));
        for(int i=0;i<N;i++){ini[i].x=nd(rng);ini[i].y=nd(rng);}
        ini[0].x=0;ini[0].y=0;
        CK(cudaMemcpy(dth,ini.data(),N*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));
    }

    void save(int frame) {
        char fn[512];
        sprintf(fn,"%s/frame_%05d.bin",od.c_str(),frame);
        std::vector<double> ho(N),hr(N),ht(N),hu(N),hv(N);
        // copy to temps before in-place IFFT
        CK(cudaMemcpy(d_tmp1,dwh,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d_tmp2,drh,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d_tmp3,dth,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d_tmp4,du1,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d_tmp5,du2,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice));
        c2r(d_tmp1,do_);c2r(d_tmp2,dr);c2r(d_tmp3,dtg);c2r(d_tmp4,du1r);c2r(d_tmp5,du2r);
        CK(cudaMemcpy(ho.data(),do_,N*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hr.data(),dr,N*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ht.data(),dtg,N*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hu.data(),du1r,N*sizeof(double),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hv.data(),du2r,N*sizeof(double),cudaMemcpyDeviceToHost));
        FILE* f=fopen(fn,"wb");
        int d[2]={p.Nx,p.Ny};
        fwrite(d,sizeof(int),2,f);
        fwrite(ho.data(),sizeof(double),N,f);
        fwrite(hr.data(),sizeof(double),N,f);
        fwrite(ht.data(),sizeof(double),N,f);
        fwrite(hu.data(),sizeof(double),N,f);
        fwrite(hv.data(),sizeof(double),N,f);
        fclose(f);
    }

    double compute_spectral_median(cufftDoubleComplex* fhat) {
        std::vector<cufftDoubleComplex> fh(N);
        CK(cudaMemcpy(fh.data(),fhat,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));
        std::vector<double> k2_h(N);
        CK(cudaMemcpy(k2_h.data(),dk2,N*sizeof(double),cudaMemcpyDeviceToHost));
        double kmax=0;
        for(int i=0;i<N;i++){double k=sqrt(k2_h[i]);if(k>kmax)kmax=k;}
        int nbins=64;
        double dk=(kmax>0)?kmax/nbins:1.0;
        std::vector<double> shell_energy(nbins,0.0);
        for(int i=0;i<N;i++){
            double k=sqrt(k2_h[i]);
            if(k>0){int bin=std::min((int)(k/dk),nbins-1);shell_energy[bin]+=fh[i].x*fh[i].x+fh[i].y*fh[i].y;}
        }
        double total=0;
        for(int b=0;b<nbins;b++)total+=shell_energy[b];
        double cumulative=0,median_k=kmax/2.0;
        for(int b=0;b<nbins;b++){
            cumulative+=shell_energy[b];
            if(cumulative>=0.5*total){median_k=(b+0.5)*dk;break;}
        }
        return median_k;
    }

    void compute_spectral_distribution(cufftDoubleComplex* fhat, const std::string& fname) {
        std::vector<cufftDoubleComplex> fh(N);
        CK(cudaMemcpy(fh.data(),fhat,N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost));
        std::vector<double> k2_h(N);
        CK(cudaMemcpy(k2_h.data(),dk2,N*sizeof(double),cudaMemcpyDeviceToHost));
        double kmax=0;
        for(int i=0;i<N;i++){double k=sqrt(k2_h[i]);if(k>kmax)kmax=k;}
        int nbins=64;
        double dk=(kmax>0)?kmax/nbins:1.0;
        std::vector<double> shell_energy(nbins,0.0);
        std::vector<int> shell_count(nbins,0);
        for(int i=0;i<N;i++){
            double k=sqrt(k2_h[i]);
            if(k>0){int bin=std::min((int)(k/dk),nbins-1);shell_energy[bin]+=fh[i].x*fh[i].x+fh[i].y*fh[i].y;shell_count[bin]++;}
        }
        FILE* f=fopen(fname.c_str(),"w");
        fprintf(f,"wavenumber,energy,count\n");
        for(int b=0;b<nbins;b++)fprintf(f,"%.6f,%.12e,%d\n",(b+0.5)*dk,shell_energy[b],shell_count[b]);
        fclose(f);
    }

    void run() {
        init();
        int ws=(int)(p.Tw/p.dt);
        for(int s=0;s<ws;s++){step_vort();step_scl(drh,dr);}

        // re-init tangent after warmup
        std::vector<cufftDoubleComplex> ini(N);
        std::mt19937 rng(99999);
        std::normal_distribution<double> nd(0,1);
        for(int i=0;i<N;i++){ini[i].x=nd(rng);ini[i].y=nd(rng);}
        ini[0].x=0;ini[0].y=0;
        CK(cudaMemcpy(dth,ini.data(),N*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice));

        int tot=(int)(p.Tr/p.dt);
        double t=0,ls=0,lt=0,pn=1.0;

        step_scl(dth,dtg);
        pn=cnorm(dtg);
        if(pn>1e-15){k_norm<<<G1,B1>>>(dtg,1.0/pn);r2c(dtg,dth);}
        pn=1.0;

        std::vector<double> ts,en,ens,ly,sns;
        printf("%8s %10s %12s %12s %10s\n","Step","Time","Energy","Lyap","Scalar");

        for(int s=0;s<tot;s++){
            step_vort();step_scl(drh,dr);step_scl(dth,dtg);
            t+=p.dt;
            double tn=cnorm(dtg);
            if(tn>1e-15&&pn>1e-15){ls+=log(tn/pn);lt+=p.dt;}
            if((s+1)%p.lr==0){
                if(tn>1e-15){k_norm<<<G1,B1>>>(dtg,1.0/tn);r2c(dtg,dth);}
                pn=1.0;
            }else{pn=tn;}

            if((s+1)%p.si==0){
                c2r(du1,du1r);c2r(du2,du2r);
                double e=cnorm(du1r),e2=cnorm(du2r);
                e=0.5*(e*e+e2*e2);
                double sn=cnorm(dr);
                double cl=(lt>0)?ls/lt:0;
                ts.push_back(t);en.push_back(e);ens.push_back(0);ly.push_back(cl);sns.push_back(sn);
                save((s+1)/p.si);
                printf("%8d %10.3f %12.4e %12.4e %10.4e\n",s+1,t,e,cl,sn);
            }
        }

        double fl=(lt>0)?ls/lt:0;
        double batchelor=(fabs(fl)>1e-15)?sqrt(p.kappa/fabs(fl)):0;
        double batchelor_k=(batchelor>0)?2*M_PI/batchelor:0;

        c2r(drh,dr);r2c(dr,drh);
        double smedian=compute_spectral_median(drh);
        compute_spectral_distribution(drh,od+"/spectrum.csv");

        FILE* f=fopen((od+"/timeseries.csv").c_str(),"w");
        fprintf(f,"time,energy,enstrophy,lyapunov,scalar_norm\n");
        for(size_t i=0;i<ts.size();i++)
            fprintf(f,"%.6f,%.8e,%.8e,%.8e,%.8e\n",ts[i],en[i],ens[i],ly[i],sns[i]);
        fclose(f);

        FILE* m=fopen((od+"/metadata.txt").c_str(),"w");
        fprintf(m,"Nx %d\nNy %d\nnu %.8f\nkappa %.8f\nsigma %.8f\nsave_interval %d\nnum_frames %d\nlyap %.8f\nbatchelor_scale %.8e\nbatchelor_k %.8e\nspectral_median %.8e\n",
                p.Nx,p.Ny,p.nu,p.kappa,p.sig,p.si,(int)ts.size(),fl,batchelor,batchelor_k,smedian);
        fclose(m);

        printf("lambda=%.6f chi_B=%.4e k_M=%.4f\n",fl,batchelor,smedian);
    }
};

int main(int argc, char** argv) {
    P pp;
    std::string od="output";
    for(int i=1;i<argc;i++){
        std::string a(argv[i]);
        if(a=="--Nx"&&i+1<argc)pp.Nx=atoi(argv[++i]);
        else if(a=="--Ny"&&i+1<argc)pp.Ny=atoi(argv[++i]);
        else if(a=="--nu"&&i+1<argc)pp.nu=atof(argv[++i]);
        else if(a=="--kappa"&&i+1<argc)pp.kappa=atof(argv[++i]);
        else if(a=="--sigma"&&i+1<argc)pp.sig=atof(argv[++i]);
        else if(a=="--dt"&&i+1<argc)pp.dt=atof(argv[++i]);
        else if(a=="--T"&&i+1<argc)pp.Tr=atof(argv[++i]);
        else if(a=="--si"&&i+1<argc)pp.si=atoi(argv[++i]);
        else if(a=="--lr"&&i+1<argc)pp.lr=atoi(argv[++i]);
        else if(a=="--outdir"&&i+1<argc)od=argv[++i];
    }
    system(("mkdir -p "+od).c_str());
    S s(pp,od);
    s.run();
    return 0;
}
