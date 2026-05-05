#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  float count[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_loadu_ps(x);
  __m512 yvec = _mm512_loadu_ps(y);
  __m512 mvec = _mm512_loadu_ps(m);
  __m512 zvec = _mm512_setzero_ps();
  __m512 onevec = _mm512_set1_ps(1.0);
  for(int i=0; i<N; i++) {
    __m512 setxvec = _mm512_set1_ps(x[i]);
    __m512 rxvec = _mm512_sub_ps(setxvec,xvec);
    __m512 rx2vec = _mm512_mul_ps(rxvec,rxvec);
    __m512 setyvec = _mm512_set1_ps(y[i]);
    __m512 ryvec = _mm512_sub_ps(setyvec,yvec);
    __m512 ry2vec = _mm512_mul_ps(ryvec,ryvec);
    __m512 r2vec = _mm512_add_ps(rx2vec,ry2vec);

    __mmask16 mask = (__mmask16)(1 << i);
    
    r2vec = _mm512_mask_blend_ps(mask,r2vec,onevec);
    
    __m512 invr2vec = _mm512_rcp14_ps(r2vec);
    __m512 invrvec = _mm512_rsqrt14_ps(r2vec);
    __m512 denomvec = _mm512_mul_ps(invr2vec,invrvec);

    denomvec = _mm512_mask_blend_ps(mask,denomvec,zvec);
    
    __m512 numerfxvec = _mm512_mul_ps(rxvec,mvec);
    __m512 numerfyvec = _mm512_mul_ps(ryvec,mvec);

    __m512 fxvec = _mm512_mul_ps(numerfxvec,denomvec);
    __m512 fyvec = _mm512_mul_ps(numerfyvec,denomvec);

    fx[i] = -1*(_mm512_reduce_add_ps(fxvec));
    fy[i] = -1*(_mm512_reduce_add_ps(fyvec));
    
    /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    */
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
