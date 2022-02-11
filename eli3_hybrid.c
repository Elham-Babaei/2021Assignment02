
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
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
 

void swap(struct kd_node_t *x, struct kd_node_t *y) {
    float_t tmp[NDIM];
    memcpy(tmp,  x->x, sizeof(tmp));
    memcpy(x->x, y->x, sizeof(tmp));
    memcpy(y->x, tmp,  sizeof(tmp));
}
// a function to find the median of datapoints
struct kd_node_t* find_median(struct kd_node_t *start, struct kd_node_t *end, int idx)
{
    if (end <= start) return NULL;
    if (end == start + 1)
        return start;
 
    struct kd_node_t *p, *store, *md = start + (end - start) / 2;
    float_t pivot;
    while (1) {
        pivot = md->x[idx];
 
        swap(md, end - 1);
        for (store = p = start; p < end; p++) {
            if (p->x[idx] < pivot) {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);
 
        /* median has duplicate values */
        if (store->x[idx] == md->x[idx])
            return md;
 
        if (store > md) end = store;
        else        start = store;
    }
}
 
// function to get the minimum of an array
    float_t getmin(float_t a[], int arraysize){
        float_t min = a[0];
        for (int i=0; i<arraysize; i++){
            if (a[i] < min) min=a[i];   
        }
    return min;
}

// function to get the maximum of an array
    float_t getmax(float_t a[], int arraysize){
        float_t max= a[0];
        for (int i=0; i<arraysize; i++){
            if (a[i] > max) max=a[i];   
        }
    return max;
}



// Creating the kdtree
    struct kd_node_t* make_tree(struct kd_node_t *points, int len, int index, int ndim, int rank)   //index= first splitting indext , ndim=2 , len= the number of points N
    {
    struct kd_node_t *n;   
    if (!len) return 0;
    
    // seperating X and Y axises
    float_t X[len];
    float_t Y[len];
    for (int i=0; i<len; ++i){
        for (int j=0; j<2; ++j){
            if(j==0)  X[i]=  points[i].x[j];
            else  Y[i]=  points[i].x[j];
        }
    } 
    
    // ** without sorting **
    int arraysize=len;
    float_t X_min = getmin(X, arraysize);
    float_t X_max = getmax(X, arraysize);
    float_t Y_min = getmin(Y, arraysize);
    float_t Y_max = getmax(Y, arraysize);
    float_t X_extent = X_max - X_min;
    float_t Y_extent = Y_max - Y_min;
    
    //print the data 
    // printf("\nX : \n");
    // for (int i=0; i<len; ++i)  printf("%f\n", X[i]);
    // printf("Y : \n");
    // for (int i=0; i<len; ++i)  printf("%f\n", Y[i]);
   
    // print the extent of each direction
    //printf("\nX_extent : %f", X_extent);
    //printf("\nY_extent : %f\n", Y_extent);
                 
    int Xlength = (sizeof(X)/ sizeof(X[0]));
    int Ylength = (sizeof(Y)/ sizeof(Y[0]));
    //printf("Xlength : %d\n", Xlength);
    //printf("Ylength : %d\n", Ylength);

    if (Xlength >1 && Ylength >1){                                  //there should be at least two points in each direction
        if ((n = find_median(points, points + len, index))) {
        int myindex=index;                                         // X is the splitting dim 0
        if (Y_extent > X_extent)  myindex= (index + 1) % ndim;     // Y is the splitting dim 1                                         
        n->index = myindex; 
        //printf("The splitting index : %d\n", myindex);
        //printf("The spliting point (x,y) : (%f, %f)\n\n", n->x[0], n->x[1]);
        
        #pragma omp parallel 
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    n->left  = make_tree(points, n - points, myindex, ndim);
                    int my_thread_id_left = omp_get_thread_num();
                    printf("mpi processor %d, openmp thread %d creating left tree\n", rank, my_thread_id_left);
                }
                #pragma omp task
                {
                    n->right = make_tree(n + 1, points + len - (n + 1), myindex, ndim, rank);
                    int my_thread_id_right = omp_get_thread_num();
                    printf("mpi processor %d, openmp thread %d creating left tree\n", rank, my_thread_id_right);
                }
            }
        }
      
    


       }
    }
    return n;
}





int main(int argc, char* argv[])
{
    struct kd_node_t  *mytree, *mypoints;
    int size;                         // the number of processors in the communicator
    int rank;
    double start_time, end_time;    //to record the execution time on each processor
   

    // the type of mypoits is kd_node_t*
    mypoints =(struct kd_node_t*) calloc(N, sizeof(struct kd_node_t));    //initializes an array of size N with values {0,0}
    for (int i = 0; i < N; i++) rand_pt(mypoints[i]);                         // filling in the array with random elemnts
    
    // start MPI
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );   // default communicator
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 
    start_time = MPI_Wtime();

    mytree = make_tree(mypoints, N, 0, 2, rank);
    
    end_time = MPI_Wtime();
    printf ( "\n # walltime on processor %i : %10.8f \n",rank, end_time - start_time ) ;
    MPI_Finalize( );

    // free(mytree);
    return 0;
}
