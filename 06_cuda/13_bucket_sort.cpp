#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucketSize(int *d_key, int *d_bucket, int N){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<N){
    int value = d_key[idx];
    atomicAdd(&d_bucket[value],1);
  }
}

__global__ void reconstruct(int *d_key, int *d_bucket_prefix, int N){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<N){
    for (int i=d_bucket_prefix[idx];i<d_bucket_prefix[idx+1];i++){
      d_key[i] = idx;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key,n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket,range*sizeof(int));
  cudaMemset(bucket, 0, range * sizeof(int));
  bucketSize<<<1,n>>>(key,bucket,n);
  cudaDeviceSynchronize();

  int *bucketPrefix;
  cudaMallocManaged(&bucketPrefix,(range+1)*sizeof(int));
  bucketPrefix[0]=0;
  for(int i=0;i<range;i++){
    bucketPrefix[i+1]=bucketPrefix[i]+bucket[i];
  }
  reconstruct<<<1,range>>>(key,bucketPrefix,range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/

