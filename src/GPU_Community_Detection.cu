#include <GPU_Community_Detection.cuh>
using namespace std;

static int CD_GRAPHSIZE;
static vector<int> row_ptr, col_indices;
static vector<int> neighbor;
static int* row_ptr_gpu;
static int* labels_gpu, * neighbor_gpu;
static int* reduce_label, * reduce_label_count;
static int* updating;

template <typename T>
void make_csr(graph_structure<T> &graph, int& CD_GRAPHSIZE)
{
    CD_GRAPHSIZE = graph.size();
    row_ptr.resize(CD_GRAPHSIZE + 1);
    row_ptr[0] = 0;
    CSR_graph<T> ARRAY_graph;
    ARRAY_graph=graph.toCSR();
    row_ptr=ARRAY_graph.OUTs_Neighbor_start_pointers;
    neighbor=ARRAY_graph.OUTs_Edges;
    col_indices=neighbor;
    // for (int i = 0; i < CD_GRAPHSIZE; i++)
    // {
    //     for (auto& edge : graph.OUTs[i])
    //     {
    //         int neighbor_vertex = edge.first;
    //         neighbor.push_back(neighbor_vertex);
    //         col_indices.push_back(neighbor_vertex);
    //     }
    //     row_ptr[i + 1] = row_ptr[i] + graph.ADJs[i].size();
    // }
}


__global__ void init_label(int* labels_gpu,int CD_GRAPHSIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < CD_GRAPHSIZE)
    {
        labels_gpu[tid] = tid;
    }
}

__global__ void LPA(int* row_ptr_gpu, int* labels_gpu, int* neighbor_gpu, int* reduce_label, int* reduce_label_count,int CD_GRAPHSIZE,int BLOCK_PER_VER)
{
    extern __shared__ int label_counts[];
    extern __shared__ int label[];
    int ver = blockIdx.x / BLOCK_PER_VER;
    int tid = (blockIdx.x % BLOCK_PER_VER) * blockDim.x + threadIdx.x;
    int segment_order = blockIdx.x % BLOCK_PER_VER;
    int stid = threadIdx.x;
    if (stid == ver)
    {
        label_counts[stid] = 1;
    }
    else
    {
        label_counts[stid] = 0;
    }
    label[stid] = tid;

    __syncthreads();

    int start = row_ptr_gpu[ver], end = row_ptr_gpu[ver + 1];
    if (tid >= end - start)
    {
        return;
    }
    int neighbor_label = labels_gpu[neighbor_gpu[start + tid]];
    if (neighbor_label >= segment_order * CD_THREAD_PER_BLOCK && neighbor_label < (segment_order + 1) * CD_THREAD_PER_BLOCK)
        atomicAdd(&label_counts[neighbor_label - segment_order * CD_THREAD_PER_BLOCK], 1);

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (label_counts[tid] < label_counts[tid + s])
            {
                label_counts[tid] = label_counts[tid + s];
                label[tid] = label_counts[tid + s];
            }
            else if (label_counts[tid] == label_counts[tid + s] && label[tid] > label_counts[tid + s])
            {
                label[tid] = label_counts[tid + s];
            }
        }
        __syncthreads();
    }
    reduce_label_count[blockIdx.x] = label_counts[0];
    reduce_label[blockIdx.x] = label[0];
    return;
}

__global__ void Updating_label(int* reduce_label, int* reduce_label_count, int* updating, int* labels_gpu,int CD_GRAPHSIZE,int BLOCK_PER_VER)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= CD_GRAPHSIZE)
        return;
    int cont = 1, label = labels_gpu[tid];
    int start = tid * BLOCK_PER_VER, end = start + BLOCK_PER_VER;
    for (int i = start; i < end; ++i)
    {
        if (reduce_label_count[i] > cont)
        {
            cont = reduce_label_count[i];
            label = reduce_label[i];
        }
        else if (reduce_label_count[i] == cont && reduce_label[i] < label)
        {
            label = reduce_label[i];
        }
    }
    if (label != labels_gpu[tid])
        *updating = 1;
    labels_gpu[tid] = label;
    return;
}

int Community_Detection(graph_structure<double>& graph, float* elapsedTime)
{
    make_csr(graph,CD_GRAPHSIZE);

    int BLOCK_PER_VER=((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK);
    int REDUCE_BLOCK_PER_GRID=(CD_GRAPHSIZE * BLOCK_PER_VER + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK;

    dim3 blockPerGrid((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1);
    dim3 useBlock((CD_GRAPHSIZE + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK * CD_GRAPHSIZE, 1, 1);
    dim3 threadPerBlock(CD_THREAD_PER_BLOCK, 1, 1);
    dim3 reduceBlock(REDUCE_BLOCK_PER_GRID, 1, 1);

    cudaMalloc(&row_ptr_gpu, (CD_GRAPHSIZE + 1) * sizeof(int));
    cudaMalloc(&labels_gpu, CD_GRAPHSIZE * sizeof(int));
    cudaMalloc(&neighbor_gpu, neighbor.size() * sizeof(int));
    cudaMalloc(&reduce_label, CD_GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMalloc(&reduce_label_count, CD_GRAPHSIZE * BLOCK_PER_VER * sizeof(int));
    cudaMemcpy(row_ptr_gpu, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_gpu, neighbor.data(), neighbor.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMallocManaged(&updating, sizeof(int));
    
    int it=0;
    *updating = 1;
    init_label << <blockPerGrid, threadPerBlock >> > (labels_gpu,CD_GRAPHSIZE);
    cudaDeviceSynchronize();
    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    cudaEventRecord(GPUstart, 0);
    while (*updating)
    {
        it++;
        *updating = 0;
        LPA << <useBlock, threadPerBlock, sizeof(int)* CD_THREAD_PER_BLOCK >> > (row_ptr_gpu, labels_gpu, neighbor_gpu, reduce_label, reduce_label_count,CD_GRAPHSIZE,BLOCK_PER_VER);
        cudaDeviceSynchronize();
        Updating_label << <reduceBlock, threadPerBlock >> > (reduce_label, reduce_label_count, updating, labels_gpu,CD_GRAPHSIZE,BLOCK_PER_VER);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);


    cudaEventElapsedTime(elapsedTime, GPUstart, GPUstop);

    cudaEventDestroy(GPUstart);
    cudaEventDestroy(GPUstop);

    cudaFree(row_ptr_gpu);
    cudaFree(labels_gpu);
    cudaFree(neighbor_gpu);
    cudaFree(reduce_label);
    cudaFree(reduce_label_count);

    return 0;
}
