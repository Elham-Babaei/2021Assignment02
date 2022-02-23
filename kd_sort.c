
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>


// #if defined(_OPENMP)
// #define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
// 		  (double)ts.tv_nsec * 1e-9)

// #define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
// 		     (double)myts.tv_nsec * 1e-9)

// #else

// #define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
// 		  (double)ts.tv_nsec * 1e-9)
// #endif


#if !defined(DOUBLE_PRECISION)
#define float_t float
#else
#define float_t double
#endif
#define NDIM 2

#define N 10        //the number of the nodes to build kdtree
#define rand1() (rand() / (float_t)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); }        //uniformly distributed random points


struct kd_node_t{
    float_t x[NDIM];        //an array of size 2
    //float_t split[NDIM];    // the splitting element
    int index;              // the splitting dimension
    struct kd_node_t *left, *right;
};

typedef int (compare_t)(const void*, const void*, int);
extern inline compare_t compare_ge;

inline int compare_ge( const void *A, const void *B, int axis )
{
  struct kd_node_t *a = (struct kd_node_t*)A;
  struct kd_node_t *b = (struct kd_node_t*)B;

  return (a->x[axis] >= b->x[axis]);
}
 
#define SWAP(A,B,SIZE) do {int sz = (SIZE); char *a = (A); char *b = (B); \
    do { char _temp = *a;*a++ = *b;*b++ = _temp;} while (--sz);} while (0)



inline int partitioning( struct kd_node_t *data, int start, int end, compare_t cmp_ge, int axis )
{ 
  void *pivot = (void*)&data[end];
  
  int pointbreak = end-1;
  for ( int i = start; i <= pointbreak; i++ )
    if( cmp_ge( (void*)&data[i], pivot, axis ) )
      {
	while( (pointbreak > i) && cmp_ge( (void*)&data[pointbreak], pivot, axis ) ) pointbreak--;
	if (pointbreak > i ) 
	  SWAP( (void*)&data[i], (void*)&data[pointbreak--], sizeof(struct kd_node_t) );
      }  
  pointbreak += !cmp_ge( (void*)&data[pointbreak], pivot, axis ) ;
  SWAP( (void*)&data[pointbreak], pivot, sizeof(struct kd_node_t) );
  
  return pointbreak;
}


void pqsort( struct kd_node_t *data, int start, int end, compare_t cmp_ge, int axis)
{
  int size = end-start;
  if ( size > 2 )
    {
      int mid = partitioning( data, start, end, cmp_ge, axis );
      
     #pragma omp task shared(data) firstprivate(start, mid)
      pqsort( data, start, mid, cmp_ge,axis );
     #pragma omp task shared(data) firstprivate(mid, end)
      pqsort( data, mid+1, end , cmp_ge,axis );
    }
  else
    {
      if ( (size == 2) && cmp_ge ( (void*)&data[start], (void*)&data[end-1], axis ) )
	SWAP( (void*)&data[start], (void*)&data[end-1], sizeof(struct kd_node_t) );
    }
    printf("hello");
}





int main(int argc, char* argv[])
{
    struct kd_node_t   *mypoints;

    mypoints =(struct kd_node_t*) calloc(N, sizeof(struct kd_node_t));    //initializes an array of size N with values {0,0}
    for (int i = 0; i < N; i++) rand_pt(mypoints[i]);                         // filling in the array with random elemnts
    

   pqsort( mypoints, 0, N, compare_ge, 0);

    // seperating X and Y axises
    float_t X[N];
    float_t Y[N];
    for (int i=0; i<N; ++i){
        for (int j=0; j<2; ++j){
            if(j==0)  X[i]=  mypoints[i].x[j];
            else  Y[i]=  mypoints[i].x[j];
        }
    } 

    //print the data 
    printf("\nX : \n");
    for (int i=0; i<N; ++i)  printf("%f\n", X[i]);
    printf("Y : \n");
    for (int i=0; i<N; ++i)  printf("%f\n", Y[i]);

    

    free(mypoints);
    return 0;
}
