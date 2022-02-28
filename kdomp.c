
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>


#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
		  (double)ts.tv_nsec * 1e-9)

#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
		     (double)myts.tv_nsec * 1e-9)

#else

#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
		  (double)ts.tv_nsec * 1e-9)
#endif


#if !defined(DOUBLE_PRECISION)
#define float_t float
#else
#define float_t double
#endif
#define NDIM 2

#define N 10000        //the number of the nodes to build kdtree
#define rand1() (rand() / (float_t)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); }        //uniformly distributed random points


struct kd_node_t{
    float_t x[NDIM];        //an array of size 2
    //float_t split[NDIM];    // the splitting element
    int axis;              // the splitting dimension
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
 
        // median has duplicate values 
        if (store->x[idx] == md->x[idx])
            return md;
 
        if (store > md) end = store;
        else        start = store;
    }
}
 



// Creating the kdtree
    struct kd_node_t* make_tree(struct kd_node_t *points, int len, int axis, int ndim)   //index= first splitting indext , ndim=2 , len= the number of points N
    {
    struct kd_node_t *n;   
    if (!len) return 0;
    
    //printf("The number of datapoint is:  %d\n" , len);

    if (len >1 ){                                //there should be at least two points in each direction
        if ((n = find_median(points, points + len, axis))) {
        int myaxis= (axis + 1) % ndim;                              // round robin                                   
        n->axis = myaxis; 
        //printf("The splitting index : %d\n", myindex);
        //printf("The spliting point (x,y) : (%f, %f)\n\n", n->x[0], n->x[1]);
        
    
        #pragma omp task firstprivate(myaxis,ndim) shared(n, points)
        {
           int my_thread_id_left = omp_get_thread_num();
           n->left  = make_tree(points, n - points, myaxis, ndim);
           //printf("I am thread %d creating left tree\n", my_thread_id_left);
           
        }
        #pragma omp task firstprivate(myaxis, ndim) shared(n, points)
        {
           int my_thread_id_right = omp_get_thread_num();
           n->right = make_tree(n + 1, points + len - (n + 1), myaxis, ndim);
           //printf("I am thread %d creating right tree\n", my_thread_id_right);
           
        }
    
       }
    }
    return n;
}



int main(int argc, char* argv[])
{
    struct kd_node_t  *mytree, *mypoints;
   
    double start_time, end_time;    //to record the execution time on each processor
    struct  timespec ts, myts;
    int nthreads_parallel;
   
    mypoints =(struct kd_node_t*) malloc(N * sizeof(struct kd_node_t));
    for (int i = 0; i < N; i++) rand_pt(mypoints[i]);           
    
    double tstart = CPU_TIME;

    #pragma omp parallel 
    {
        #pragma omp master
        nthreads_parallel= omp_get_num_threads();        //returns the number of threads active in that region
        
        #pragma omp single nowait                        // other threads skip the single region
        {                                                //implied barrier that makes all tasks to be completed before passing
            mytree = make_tree(mypoints, N, 0, 2);
        }  
    }

    
   printf("The number of threads entered the parallel region: %d\n", nthreads_parallel);


   double tend = CPU_TIME;
   printf("\nRun took %g of wall-clock time\n\n", tend - tstart ); 
   
   
   free(mypoints);
   
   
    return 0;
}
