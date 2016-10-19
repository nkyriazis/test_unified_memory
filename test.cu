#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>
#include <iostream>

template < typename allocator_ >
struct unified_memory_allocator
    : public allocator_
{
    using this_type = unified_memory_allocator<allocator_>;
    using allocator_type = allocator_;
    using base_type = allocator_;
    using pointer = typename base_type::pointer;
    using value_type = typename base_type::value_type;
    
    template < typename T >
    struct rebind { using other = unified_memory_allocator<typename allocator_type::template rebind<T>::other>; };

    unified_memory_allocator() {}

    template < typename T >
    unified_memory_allocator(const T &t)
        : base_type(t)
    {}

    pointer allocate(std::ptrdiff_t nelements)
    {
        void *ptr;
        auto sz = nelements * sizeof(value_type);
        cudaError err = cudaMallocManaged(&ptr, sz);
        cudaDeviceSynchronize();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        if (ptr == nullptr) throw std::runtime_error("null pointer from allocation");
        return pointer(ptr);
    }

    void deallocate(pointer ptr, size_t nelements)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

struct foo { int intVal; double dblVal; };

int main()
{
    try
    {
        std::vector<foo, unified_memory_allocator<std::allocator<foo>>> vec(10);

        auto first = vec.data();
        auto last = first + vec.size();

        thrust::for_each(thrust::cpp::par,
            first, last, [](foo &f) { f.intVal = 0; f.dblVal = 1; });

        thrust::for_each(thrust::cuda::par,
            first, last, [] __device__ (foo &f) { f.intVal = 2; f.dblVal = 3; });
        cudaThreadSynchronize();

        thrust::for_each(thrust::cpp::par,
            first, last,
            [](const foo &f)
            {
                std::cout << "int : " << f.intVal << std::endl
                    << "dbl : " << f.dblVal << std::endl;
            }
        );
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}