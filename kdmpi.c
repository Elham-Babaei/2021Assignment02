
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>




#if !defined(DOUBLE_PRECISION)
#define float_t float
#else
#define float_t double
#endif
#define NDIM 2

#define N 100000000        //the number of the nodes to build kdtree
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
    struct kd_node_t* make_tree(struct kd_node_t *points, int len, int axis, int ndim, int rank)   //index= first splitting indext , ndim=2 , len= the number of points N
    {
    struct kd_node_t *n;   
    if (!len) return 0;
   
 
    //printf("The number of datapoint is:  %d\n" , len);

    if (len >1){                                                     //there should be at least two points in each direction
        if ((n = find_median(points, points + len, axis))) {
        int myaxis= (axis + 1) % ndim;                              // round robin                                     
        n->axis = myaxis; 
        //printf("The splitting index : %d\n", myindex);
        //printf("The spliting point (x,y) : (%f, %f)\n\n", n->x[0], n->x[1]);
        
        
        n->left  = make_tree(points, n - points, myaxis, ndim, rank);
        //printf("mpi processor %d creating left tree\n", rank);
          
        n->right = make_tree(n + 1, points + len - (n + 1), myaxis, ndim, rank);
        //printf("mpi processor %d creating right tree\n", rank); 
        

       }
    }
    return n;
}




int main(int argc, char* argv[])
{
    
    int size;                         // the number of tasks in the communicator
    int rank;
    double start_time, end_time;    //to record the execution time on each processor
   
    // start MPI
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );   // default communicator
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 
    struct kd_node_t  *mytree, *mypoints, *mychunk;
    int sndcount = N/size;         // sendcount and receivecount for scattering and gathering data on tasks
    int rcvcount = N/size;

    //initializes an array of size N with values {0,0} on all tasks
    mypoints =(struct kd_node_t*) malloc(N * sizeof(struct kd_node_t));
    //chunks of size N/size on all tasks  
    mychunk = (struct kd_node_t*) malloc(N/size * sizeof(struct kd_node_t));
    
    start_time = MPI_Wtime();
    // filling in the array with random elemnts on task 0 only
    if (rank == 0){
        for (int i = 0; i < N; i++) rand_pt(mypoints[i]);           
    }
    
    // scatter the chunks
    MPI_Scatter (mypoints, sndcount, MPI_BYTE, mychunk  , rcvcount,  MPI_BYTE, 0, MPI_COMM_WORLD); 

    //create sub-trees using the chunk on all tasks
    mytree = make_tree(mychunk, N/size, 0, 2, rank);

    
    end_time = MPI_Wtime();
    printf ( "\n # walltime on processor %i : %10.8f \n",rank, end_time - start_time ) ;
    //printf("sendcound %d", sndcount);

    
    free(mypoints);
    free(mychunk);
   
    MPI_Finalize( );
    

    return 0;
}
