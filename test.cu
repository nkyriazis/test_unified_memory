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

int main()
{
    try
    {
		auto size = 10;
        std::vector<int, unified_memory_allocator<std::allocator<int>>> vec(size), target(size);

        auto first = vec.data();
        auto last = first + vec.size();
		auto targetFirst = target.data();
		auto targetLast = targetFirst + target.size();

		thrust::sequence(thrust::cpp::par, first, last);

		auto every_second = thrust::make_permutation_iterator(first,
			thrust::make_transform_iterator(thrust::make_counting_iterator(0),
				[]__device__(int i) { return 2 * i; }));
		thrust::sequence(thrust::cuda::par, every_second, every_second + 5);

		thrust::transform(thrust::cuda::par,
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(10),
			targetFirst,
			[=]__device__(int i)
			{
				return *(first + size - i - 1);
			}
		);
		cudaThreadSynchronize();

		thrust::copy(thrust::cpp::par, targetFirst, targetLast,
			std::ostream_iterator<int>(std::cout, " "));
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    return 0;
}